#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
# Developed with the assistance of the Gemini AI code assistant.

import gzip
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import requests
import xml.etree.ElementTree as ET

# Binary and directory definitions
WGRIB2_BIN = "/apps/wgrib2/3.1.2/gnu_13.2.0/wmo/bin/wgrib2"
OUTPUT_BASE_DIR = Path("mrms_501_251")
S3_BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"

def list_s3_files(prefix: str):
    """Fetch file listing from NOAA MRMS public S3 bucket XML interface."""
    # Fixed: Removed trailing slash at the end of prefix parameter
    url = f"{S3_BUCKET_URL}/?prefix={prefix}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return []

        # Parse XML response from S3
        root = ET.fromstring(response.content)
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        keys = [elem.text for elem in root.findall('.//s3:Key', ns)]
        return keys
    except Exception as e:
        print(f"Error listing S3 path {prefix}: {e}")
        return []

def generate_date_range(start_date: datetime, end_date: datetime):
    """Generate daily timestamps across specified interval."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)

def main():
    # Date range configuration
    start_date = datetime(2026, 3, 1)
    end_date = datetime(2026, 9, 30)

    for current_date in generate_date_range(start_date, end_date):
        date_str = current_date.strftime("%Y%m%d")
        day_dir = OUTPUT_BASE_DIR / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        print(f"Listing MRMS objects for {date_str}...")

        # Formulate prefix matching S3 path structure
        # Strategy 1: Subdirectory path (CONUS/MRMS_MergedReflectivityQCComposite_00.50/YYYYMMDD/)
        prefix = f"CONUS/MRMS_MergedReflectivityQCComposite_00.50/{date_str}/"
        s3_keys = list_s3_files(prefix)

        # Fallback Strategy 2: Pre-2020 path structure without "MRMS_" prefix or flat file structure
        if not s3_keys:
            prefix = f"CONUS/MergedReflectivityQCComposite_00.50/{date_str}/"
            s3_keys = list_s3_files(prefix)

        if not s3_keys:
            # Fallback Strategy 3: Flat filename prefix match
            prefix = f"CONUS/MRMS_MergedReflectivityQCComposite_00.50_{date_str}"
            s3_keys = list_s3_files(prefix)

        if not s3_keys:
            print(f"⚠️ No objects found on S3 for date {date_str}")
            continue

        print(f"Found {len(s3_keys)} key(s) for {date_str}")

        for key in s3_keys:
            filename = Path(key).name

            # Match top-of-hour files: -HH0000.grib2 or -HH0000.grib2.gz
            if not re.search(r'-\d{2}00\d{2}\.grib2(\.gz)?$', filename):
                continue

            # Target output file path
            uncompressed_name = filename.replace('.gz', '')
            output_grib = day_dir / uncompressed_name

            # Skip existing valid files
            if output_grib.exists() and output_grib.stat().st_size > 200000:
                print(f"Skipping existing: {output_grib}")
                continue

            file_url = f"{S3_BUCKET_URL}/{key}"
            print(f"Downloading & regridding: {filename}...")

            try:
                response = requests.get(file_url, timeout=30)
                if response.status_code != 200:
                    print(f"Failed to fetch {file_url}")
                    continue

                # Decompress if gzipped; otherwise use raw content
                if filename.endswith('.gz'):
                    grib_data = gzip.decompress(response.content)
                else:
                    grib_data = response.content

                # Run wgrib2 regrid via stdin
                cmd = [
                    WGRIB2_BIN, "-",
                    "-new_grid_winds", "earth",
                    "-new_grid", "latlon", "-122.0:501:0.1", "25.00:251:0.1",
                    str(output_grib)
                ]

                proc = subprocess.run(
                    cmd,
                    input=grib_data,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                print(f"✅ Successfully created: {output_grib}")

            except subprocess.CalledProcessError as e:
                print(f"❌ wgrib2 error processing {filename}: {e.stderr.decode()}")
            except Exception as e:
                print(f"❌ Failed processing {filename}: {e}")

if __name__ == "__main__":
    main()
