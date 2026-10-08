#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
# Developed with the assistance of the Gemini AI code assistant.
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Binary and output configuration
WGRIB2_BIN = "wgrib2"
OUTPUT_BASE_DIR = Path("hrrr_501_251")

# AWS S3 HRRR Base URL
AWS_BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

# Shared environment configuration for wgrib2 dynamic library linking
ENV = os.environ.copy()
LIB_PATHS = [
    "/apps/wgrib2/3.1.2/gnu_13.2.0/lib",
    "/apps/wgrib2/3.1.2/gnu_13.2.0/ncep/lib",
    "/apps/wgrib2/3.1.2/gnu_13.2.0/wmo/lib",
    "/apps/netcdf/lib",
    "/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/lib",
    "/usr/local/lib",
    "/usr/lib64"
]
existing_ld = ENV.get("LD_LIBRARY_PATH", "")
valid_paths = [p for p in LIB_PATHS if os.path.exists(p)]
ENV["LD_LIBRARY_PATH"] = ":".join(valid_paths) + (f":{existing_ld}" if existing_ld else "")

# Filter regex for wgrib2 slice
WGRIB_FILTER = (
    ":SPFH:850 mb|:SPFH:2 m above ground|:PWAT|:UGRD:600 mb|:VGRD:600 mb|"
    ":UGRD:10 m above ground|:VGRD:10 m above ground|:UGRD:800 mb|:VGRD:800 mb|"
    ":UGRD:850 mb|:VGRD:850 mb|:MSLMA:mean sea level|:RH:850 mb|:CAPE:surface|"
    ":CIN:surface|:VVEL:500|:VVEL:700|:VVEL:925|:LFTX:500-1000 mb|:REFC|"
    ":RH:1013.2 mb|:TMP:surface|:PRES:surface|:SOILW:0-0 m below ground|:DPT:2 m above ground"
)


def get_robust_session() -> requests.Session:
    """Create a requests session with robust retry logic for cloud streaming."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (Python/HRRR-Downloader)"})
    return session


def download_file(session: requests.Session, url: str, output_path: Path) -> bool:
    """Download GRIB file safely with connection error handling."""
    try:
        with session.get(url, stream=True, timeout=60) as response:
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                return True
            else:
                print(f"HTTP {response.status_code} when downloading {url}")
                return False
    except Exception as e:
        print(f"Stream error for {url}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def generate_date_range(start_date: datetime, end_date: datetime):
    """Yield dates continuously between start_date and end_date."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def main():
    start_date = datetime(2022, 3, 1)
    end_date = datetime(2022, 10, 26)

    cycles = range(0, 24)         # 00z through 23z
    forecast_hours = range(1, 9)  # f02 - f08

    session = get_robust_session()

    for current_date in generate_date_range(start_date, end_date):
        date_str = current_date.strftime("%Y%m%d")
        dir_name = OUTPUT_BASE_DIR / date_str
        dir_name.mkdir(parents=True, exist_ok=True)

        print(f"Processing Date: {date_str}")

        for cycle in cycles:
            cycle_str = f"{cycle:02d}"

            for fct_hr in forecast_hours:
                fct_str = f"{fct_hr:02d}"

                hrrr_filename = f"hrrr.t{cycle_str}z.wrfprsf{fct_str}.grib2"
                aws_url = f"{AWS_BASE_URL}/hrrr.{date_str}/conus/{hrrr_filename}"
                #if int(date_str) != 2021 and int(date_str) != 2022: continue
                temp_grib = Path(f"temp_{date_str}_{cycle_str}_{fct_str}.grib2")
                output_grib = dir_name / f"{date_str}_t{cycle_str}zprsf{fct_str}.grib2"
                print(output_grib)

                # Skip download/processing if final file exists and is valid size
                if output_grib.exists() and output_grib.stat().st_size > 350000:
                    print(f"Skipping existing file: {output_grib}")
                    continue

                try:
                    print(f"Fetching {hrrr_filename} from AWS S3...")
                    success = download_file(session, aws_url, temp_grib)

                    if success and temp_grib.exists():
                        # Construct wgrib2 processing pipeline
                        p1 = subprocess.Popen([WGRIB2_BIN, str(temp_grib), "-s"], stdout=subprocess.PIPE, env=ENV)
                        p2 = subprocess.Popen(["egrep", WGRIB_FILTER], stdin=p1.stdout, stdout=subprocess.PIPE)
                        p1.stdout.close()

                        cmd_regrid = [
                            WGRIB2_BIN, "-i", str(temp_grib),
                            "-new_grid_winds", "earth",
                            "-new_grid", "latlon", "-122.0:501:0.1", "25.00:251:0.1",
                            str(output_grib)
                        ]

                        subprocess.run(cmd_regrid, stdin=p2.stdout, env=ENV, check=True)
                        p2.stdout.close()
                        print(f"Successfully processed: {output_grib}")
                    else:
                        print(f"Could not download {hrrr_filename} from AWS S3.")

                except Exception as e:
                    print(f"Failed processing cycle {cycle_str} f{fct_str} on {date_str}: {e}")

                finally:
                    # Clean up temporary full-sized GRIB file
                    if temp_grib.exists():
                        temp_grib.unlink()

if __name__ == "__main__":
    main()
