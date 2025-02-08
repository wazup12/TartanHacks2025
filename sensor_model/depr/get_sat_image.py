#!/usr/bin/env python3
import requests
import argparse

def download_satellite_image(lat, lon, zoom, output_file):
    """
    Downloads a 500x500 satellite image from Google Static Maps centered at the specified coordinate.

    Parameters:
        lat (float): Latitude of the center point.
        lon (float): Longitude of the center point.
        zoom (int): Zoom level (typical values range from 0 (world) to 21 (building)).
        api_key (str): Your Google Maps API key.
        output_file (str): Filename to save the downloaded image.
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": str(zoom),
        "size": "500x500",
        "maptype": "satellite",  # pure aerial imagery without additional labels
        "key": "AIzaSyAAQfRZTVRl8VpxpRfKtbL_phKc1PLwYQs",
        # If you want even higher quality (i.e. a higher resolution image), you can uncomment:
        # "scale": "2"
    }

    print("Requesting image...")
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Image successfully saved to '{output_file}'.")
    else:
        print("Error downloading image:")
        print(f"Status code: {response.status_code}")
        print(response.text)

def main():
    # parser = argparse.ArgumentParser(
    #     description="Download a 500x500 high-quality satellite image from a specified coordinate using Google Static Maps API."
    # )
    # parser.add_argument("--lat", type=float, required=True, help="Latitude of the center point.")
    # parser.add_argument("--lon", type=float, required=True, help="Longitude of the center point.")
    # parser.add_argument("--zoom", type=int, default=18, help="Zoom level (default is 18).")
    # parser.add_argument("--key", type=str, required=True, help="Your Google Maps API key.")
    # parser.add_argument("--output", type=str, default="satellite_image.png", help="Output filename (default is satellite_image.png).")
    # args = parser.parse_args()

    # download_satellite_image(args.lat, args.lon, args.zoom, args.output)
    lat = 40.4447605
    lon = -79.9426024
    zoom = 18
    file_name = "test1.jpg"
    download_satellite_image(lat, lon, zoom, file_name)

if __name__ == "__main__":
    main()
