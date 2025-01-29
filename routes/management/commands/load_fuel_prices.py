from django.core.management.base import BaseCommand
import csv
from pathlib import Path
from routes.models import FuelStation
from geopy.geocoders import Nominatim
import time
from geopy.exc import GeocoderTimedOut

class Command(BaseCommand):
    help = 'Load fuel prices from CSV'

    def geocode_with_retry(self, geolocator, address, max_retries=3):
        for i in range(max_retries):
            try:
                return geolocator.geocode(address, timeout=5)
            except GeocoderTimedOut:
                if i == max_retries - 1:
                    raise
                time.sleep(1)

    def handle(self, *args, **kwargs):
        csv_file = Path(__file__).parent.parent.parent.parent / 'data' / 'fuel-prices-for-be-assessment.csv'
        geolocator = Nominatim(user_agent="fuel_route_app_v1")
        
        # Clear existing data
        FuelStation.objects.all().delete()
        
        success_count = 0
        error_count = 0
        
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                full_address = f"{row['Address']}, {row['City']}, {row['State']}, USA"
                
                try:
                    location = self.geocode_with_retry(geolocator, full_address)
                    if location:
                        FuelStation.objects.create(
                            location=f"{row['Truckstop Name']} - {row['City']}, {row['State']}",
                            price=float(row['Retail Price']),
                            latitude=location.latitude,
                            longitude=location.longitude
                        )
                        success_count += 1
                        self.stdout.write(self.style.SUCCESS(f"Added: {row['Truckstop Name']} in {row['City']}, {row['State']}"))
                        time.sleep(1)  # Rate limiting
                except Exception as e:
                    error_count += 1
                    self.stdout.write(self.style.WARNING(f"Skipping {row['Truckstop Name']}: {str(e)}"))

        self.stdout.write(self.style.SUCCESS(f"\nImport completed:\n- Successfully added: {success_count} stations\n- Failed: {error_count} stations"))