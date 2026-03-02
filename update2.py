#!/opt/weather_sensors/.venv/bin/python
"""
Weather Station Logger
"""

from __future__ import annotations

import glob
import math
import time
from dataclasses import dataclass

import mysql.connector

from gpiozero import Device
from gpiozero.pins.lgpio import LGPIOFactory

Device.pin_factory = LGPIOFactory()

from gpiozero import Button

from dbconfig import read_db_config
from static.heatindex import heat_index
import static.wind_direction as wind_direction


# ==========================================================
# OPTIONAL SENSOR LIBRARY IMPORTS
# ==========================================================

try:
    from adafruit_extended_bus import ExtendedI2C as I2C
except ImportError:
    I2C = None

try:
    import adafruit_sht4x
except ImportError:
    adafruit_sht4x = None

try:
    import adafruit_veml6075
except ImportError:
    adafruit_veml6075 = None

try:
    from adafruit_bme280 import basic as adafruit_bme280
except ImportError:
    adafruit_bme280 = None

try:
    import serial
    import adafruit_pm25.uart as pm25_uart
except ImportError:
    serial = None
    pm25_uart = None


# ==========================================================
# SENSOR ENABLE FLAGS
# ==========================================================

SENSORS = {
    "SHT45":   True,
    "BME280":  True,
    "VEML6075":True,
    "DS18B20": False,
    "PMS5003": True,
    "WIND":    True,
    "RAIN":    False,
}


# ==========================================================
# CONSTANTS
# ==========================================================

MEASUREMENT_SECONDS  = 60
GUST_WINDOW_SECS     = 3
WIND_PIN             = 27
RAIN_PIN             = 6

BUCKET_SIZE_MM       = 0.27494

ANEMOMETER_RADIUS_CM = 9.0
WIND_ADJUSTMENT      = 1.18

CM_IN_A_KM           = 100000.0
SECS_IN_AN_HOUR      = 3600.0


# ==========================================================
# DATA MODEL
# ==========================================================

@dataclass
class Sample:
    air_temp_c:       float = 0.0
    feels_like_c:     float = 0.0
    pressure_sea_hpa: float = 0.0
    humidity_pct:     float = 0.0
    dew_point_c:      float = 0.0
    uv_index:         float = 0.0
    wind_dir_deg:     float = 0.0
    wind_speed_kmh:   float = 0.0
    wind_gust_kmh:    float = 0.0
    rainfall_mm:      float = 0.0
    pm1:              float = 0.0
    pm25:             float = 0.0
    pm10:             float = 0.0
    aqi:              int   = 0
    aqi_category:     str   = ""


# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def compute_dew_point_c(temp_c: float, rh: float) -> float:
    if rh <= 0:
        return temp_c
    a     = 17.62
    b     = 243.12
    gamma = (a * temp_c / (b + temp_c)) + math.log(rh / 100.0)
    return round((b * gamma) / (a - gamma), 2)


def compute_wind_speed_kmh(pulse_count: int, seconds: float) -> float:
    if seconds <= 0 or pulse_count <= 0:
        return 0.0
    circumference_cm = 2.0 * math.pi * ANEMOMETER_RADIUS_CM
    rotations        = pulse_count / 2.0
    dist_km          = (circumference_cm * rotations) / CM_IN_A_KM
    km_per_sec       = dist_km / seconds
    return km_per_sec * SECS_IN_AN_HOUR * WIND_ADJUSTMENT


def compute_aqi(pm25_avg: float, pm10_avg: float) -> int:
    """
    US EPA AQI via piecewise linear interpolation.
    Expects 24-hour average concentrations (µg/m³).
    Returns the higher of the PM2.5 and PM10 sub-indices.
    """

    PM25_BREAKPOINTS = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    PM10_BREAKPOINTS = [
        (0,    54,   0,   50),
        (55,  154,  51,  100),
        (155, 254, 101,  150),
        (255, 354, 151,  200),
        (355, 424, 201,  300),
        (425, 504, 301,  400),
        (505, 604, 401,  500),
    ]

    def _sub_index(c, breakpoints):
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= c <= c_high:
                return round(
                    ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
                )
        return 500  # beyond scale

    return max(
        _sub_index(pm25_avg, PM25_BREAKPOINTS),
        _sub_index(pm10_avg, PM10_BREAKPOINTS),
    )


def aqi_category(aqi: int) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


# ==========================================================
# SENSOR READERS
# ==========================================================

def read_sht45():
    if not SENSORS["SHT45"]:
        return None, None
    try:
        i2c = I2C(1)
        sht = adafruit_sht4x.SHT4x(i2c)
        sht.mode = adafruit_sht4x.Mode.NOHEAT_HIGHPRECISION
        return sht.measurements
    except Exception as e:
        print(f"SHT45 error: {e}")
        return None, None


def read_bme280():
    if not SENSORS["BME280"]:
        return None, None, None
    try:
        i2c = I2C(6)
        bme = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=0x76)
        bme.sea_level_pressure = 1013.25
        return bme.temperature, bme.humidity, bme.pressure
    except Exception as e:
        print(f"BME280 error: {e}")
        return None, None, None


def read_veml():
    if not SENSORS["VEML6075"]:
        return None
    try:
        i2c = I2C(4)
        veml = adafruit_veml6075.VEML6075(i2c, integration_time=100)
        return float(getattr(veml, "uv_index", 0.0))
    except Exception as e:
        print(f"VEML6075 error: {e}")
        return None


def read_ds18b20():
    if not SENSORS["DS18B20"]:
        return None
    try:
        paths = glob.glob("/sys/bus/w1/devices/28-*/temperature")
        if not paths:
            return None
        with open(paths[0], "r") as f:
            return int(f.read().strip()) / 1000.0
    except Exception as e:
        print(f"DS18B20 error: {e}")
        return None


def read_pms5003():
    if not SENSORS["PMS5003"]:
        return None, None, None
    try:
        uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=2)
        pm25 = pm25_uart.PM25_UART(uart)
        data = pm25.read()
        uart.close()
        return (
            safe_float(data.get("pm10 standard")),   # PM1.0
            safe_float(data.get("pm25 standard")),   # PM2.5
            safe_float(data.get("pm100 standard")),  # PM10
        )
    except Exception as e:
        print(f"PMS5003 error: {e}")
        return None, None, None


# ==========================================================
# WIND / RAIN
# ==========================================================

def capture_wind_rain(duration_s: int):
    if not SENSORS["WIND"] and not SENSORS["RAIN"]:
        return 0, 0, 0, 0.0

    rain_counter = [0]

    wind_btn = None
    rain_btn = None

    if SENSORS["WIND"]:
        wind_btn = Button(WIND_PIN, pull_up=True, bounce_time=0.005)

    if SENSORS["RAIN"]:
        rain_btn = Button(RAIN_PIN, pull_up=True, bounce_time=0.02)

    num_buckets       = duration_s // GUST_WINDOW_SECS
    bucket_counts     = []
    total_wind_pulses = 0

    for _ in range(num_buckets):
        bucket_pulses = [0]

        def wind_tick():
            bucket_pulses[0] += 1

        def rain_tick():
            rain_counter[0] += 1

        if SENSORS["WIND"]:
            wind_btn.when_pressed = wind_tick

        if SENSORS["RAIN"]:
            rain_btn.when_pressed = rain_tick

        time.sleep(GUST_WINDOW_SECS)

        if SENSORS["WIND"]:
            wind_btn.when_pressed = None
        if SENSORS["RAIN"]:
            rain_btn.when_pressed = None

        bucket_counts.append(bucket_pulses[0])
        total_wind_pulses += bucket_pulses[0]

    wind_dir = 0.0
    if SENSORS["WIND"]:
        try:
            wind_dir = float(wind_direction.get_wind_direction())
        except Exception:
            wind_dir = 0.0

    if wind_btn:
        wind_btn.close()
    if rain_btn:
        rain_btn.close()

    peak_bucket = max(bucket_counts) if bucket_counts else 0

    return total_wind_pulses, peak_bucket, rain_counter[0], wind_dir


# ==========================================================
# DATABASE
# ==========================================================

def fetch_24h_pm_averages() -> tuple[float, float]:
    """
    Query the rolling 24-hour average of PM25 and PM10
    from the measurements table. Falls back to 0.0 on error.
    """
    try:
        cfg = read_db_config(filename="config.ini", section="mysql")
        conn = mysql.connector.connect(
            host=cfg["host"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT AVG(PM25), AVG(PM10)
            FROM measurements
            WHERE CREATED >= NOW() - INTERVAL 24 HOUR
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row and row[0] is not None:
            return safe_float(row[0]), safe_float(row[1])
        return 0.0, 0.0
    except Exception as e:
        print(f"24h PM average query failed: {e}")
        return 0.0, 0.0


def insert_mysql(sample: Sample):
    try:
        cfg = read_db_config(filename="config.ini", section="mysql")
        conn = mysql.connector.connect(
            host=cfg["host"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
            autocommit=True,
        )
        cur = conn.cursor()

        sql = """
        INSERT INTO measurements
          (CREATED, AIR_TEMP, FEELS_LIKE, PRESSURE_SEA, HUMIDITY, DEW_POINT, UV_INDEX,
           WIND_DIRECTION, WIND_SPEED, WIND_GUST, RAINFALL,
           PM1, PM25, PM10, AQI, AQI_CATEGORY)
        VALUES
          (NOW(), %s, %s, %s, %s, %s, %s,
           %s, %s, %s, %s,
           %s, %s, %s, %s, %s)
        """

        cur.execute(sql, (
            sample.air_temp_c,
            sample.feels_like_c,
            sample.pressure_sea_hpa,
            sample.humidity_pct,
            sample.dew_point_c,
            sample.uv_index,
            sample.wind_dir_deg,
            sample.wind_speed_kmh,
            sample.wind_gust_kmh,
            sample.rainfall_mm,
            sample.pm1,
            sample.pm25,
            sample.pm10,
            sample.aqi,
            sample.aqi_category,
        ))

        cur.close()
        conn.close()

    except Exception as e:
        print(f"MySQL insert failed: {e}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    # --- SHT45 ---
    _temp, _rh = read_sht45()
    temp = safe_float(_temp)
    rh   = safe_float(_rh)

    # --- BME280 ---
    bme_temp, bme_rh, bme_pressure = read_bme280()
    pressure = safe_float(bme_pressure)

    # SHT45 preferred for temp/rh, BME280 as fallback
    if temp == 0.0 and bme_temp is not None:
        temp = safe_float(bme_temp)
    if rh == 0.0 and bme_rh is not None:
        rh = safe_float(bme_rh)

    # --- UV ---
    uv = safe_float(read_veml())

    # --- DS18B20 overrides air temp if available ---
    probe = read_ds18b20()
    if probe is not None:
        temp = safe_float(probe)

    # --- Air Quality ---
    _pm1, _pm25, _pm10 = read_pms5003()

    # --- 24h rolling average for AQI (queried before this insert) ---
    avg_pm25, avg_pm10 = fetch_24h_pm_averages()

    # If no history yet fall back to current reading
    if avg_pm25 == 0.0 and _pm25:
        avg_pm25 = safe_float(_pm25)
    if avg_pm10 == 0.0 and _pm10:
        avg_pm10 = safe_float(_pm10)

    _aqi = compute_aqi(avg_pm25, avg_pm10)
    _aqi_category = aqi_category(_aqi)

    # --- Wind / Rain ---
    wind_pulses, peak_bucket, rain_tips, wind_dir = capture_wind_rain(MEASUREMENT_SECONDS)

    wind_speed  = compute_wind_speed_kmh(wind_pulses, MEASUREMENT_SECONDS)
    wind_gust   = compute_wind_speed_kmh(peak_bucket, GUST_WINDOW_SECS)
    wind_gust   = max(wind_gust, wind_speed)
    rainfall_mm = rain_tips * BUCKET_SIZE_MM

    # --- Derived ---
    dew   = compute_dew_point_c(temp, rh)
    feels = heat_index(temp, rh) if rh > 0 else temp

    sample = Sample(
        air_temp_c       = round(temp, 2),
        feels_like_c     = round(feels, 2),
        pressure_sea_hpa = round(pressure, 2),
        humidity_pct     = round(rh, 2),
        dew_point_c      = round(dew, 2),
        uv_index         = round(uv, 2),
        wind_dir_deg     = round(wind_dir, 1),
        wind_speed_kmh   = round(wind_speed, 2),
        wind_gust_kmh    = round(wind_gust, 2),
        rainfall_mm      = round(rainfall_mm, 2),
        pm1              = round(safe_float(_pm1), 1),
        pm25             = round(safe_float(_pm25), 1),
        pm10             = round(safe_float(_pm10), 1),
        aqi              = _aqi,
        aqi_category     = _aqi_category,
    )

    insert_mysql(sample)

    # --- Print all values being written to MySQL ---
    print(f"\n{'='*44}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}  →  MySQL insert")
    print(f"{'='*44}")
    print(f"  AIR_TEMP       {sample.air_temp_c:>10.2f} °C")
    print(f"  FEELS_LIKE     {sample.feels_like_c:>10.2f} °C")
    print(f"  PRESSURE_SEA   {sample.pressure_sea_hpa:>10.2f} hPa")
    print(f"  HUMIDITY       {sample.humidity_pct:>10.2f} %")
    print(f"  DEW_POINT      {sample.dew_point_c:>10.2f} °C")
    print(f"  UV_INDEX       {sample.uv_index:>10.2f}")
    print(f"  WIND_DIRECTION {sample.wind_dir_deg:>10.1f} °")
    print(f"  WIND_SPEED     {sample.wind_speed_kmh:>10.2f} km/h")
    print(f"  WIND_GUST      {sample.wind_gust_kmh:>10.2f} km/h")
    print(f"  RAINFALL       {sample.rainfall_mm:>10.2f} mm")
    print(f"  PM1            {sample.pm1:>10.1f} µg/m³")
    print(f"  PM2.5          {sample.pm25:>10.1f} µg/m³")
    print(f"  PM10           {sample.pm10:>10.1f} µg/m³")
    print(f"  PM2.5 24h avg  {avg_pm25:>10.1f} µg/m³")
    print(f"  PM10  24h avg  {avg_pm10:>10.1f} µg/m³")
    print(f"  AQI            {sample.aqi:>10d}")
    print(f"  AQI_CATEGORY   {sample.aqi_category}")
    print(f"{'='*44}\n")


if __name__ == "__main__":
    main()