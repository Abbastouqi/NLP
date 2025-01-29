from .models import FuelStation
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import requests

class RouteCalculator:
    def __init__(self):
        self.VEHICLE_RANGE = 500  # miles
        self.MPG = 10  # miles per gallon
        self.geolocator = Nominatim(user_agent="fuel_route_app")

    def get_coordinates(self, location):
        location_data = self.geolocator.geocode(f"{location}, USA")
        if not location_data:
            raise ValueError(f"Location not found: {location}")
        return (location_data.latitude, location_data.longitude)

    def find_optimal_fuel_stops(self, start_coords, end_coords):
        total_distance = geodesic(start_coords, end_coords).miles
        num_stops = int(total_distance / self.VEHICLE_RANGE)
        
        fuel_stops = []
        total_cost = 0
        current_coords = start_coords
        used_stations = set()  # Track used stations

        for i in range(num_stops + 1):
            fraction = (i + 1) * self.VEHICLE_RANGE / total_distance
            if fraction > 1:
                fraction = 1
                
            target_lat = start_coords[0] + (end_coords[0] - start_coords[0]) * fraction
            target_lon = start_coords[1] + (end_coords[1] - start_coords[1]) * fraction
            target_coords = (target_lat, target_lon)
            
            # Find nearest station not already used
            station = self._find_optimal_station(target_coords, used_stations)
            if station:
                used_stations.add(station.location)  # Mark station as used
                fuel_stops.append({
                    'location': station.location,
                    'price': station.price,
                    'coords': [station.latitude, station.longitude]
                })
                
                gallons_needed = self.VEHICLE_RANGE / self.MPG
                total_cost += gallons_needed * station.price
                
        return fuel_stops, total_cost

    def _find_optimal_station(self, current_coords, used_stations):
        stations = FuelStation.objects.all()
        best_station = None
        best_score = float('inf')

        for station in stations:
            if station.location in used_stations:
                continue
            
            station_coords = (station.latitude, station.longitude)
            distance = geodesic(current_coords, station_coords).miles
            
            # Consider stations within reasonable range (500 miles)
            if distance <= 500:
                # Score based on price and distance (lower is better)
                score = station.price + (distance / 100)
                if score < best_score:
                    best_score = score
                    best_station = station
        
        return best_station
    def get_route_map(self, start_coords, end_coords, fuel_stops):
        # Using OpenStreetMap for demonstration
        # In production, use a service like Google Maps or Mapbox
        markers = f"markers={start_coords[0]},{start_coords[1]}"
        
        for stop in fuel_stops:
            markers += f"|{stop['coords'][0]},{stop['coords'][1]}"
            
        markers += f"|{end_coords[0]},{end_coords[1]}"
        
        return f"https://static-maps.openstreetmap.org/v1/route?{markers}"