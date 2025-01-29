from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import RouteCalculator
import logging

logger = logging.getLogger(__name__)

class RouteAPIView(APIView):
    def post(self, request):
        try:
            start_location = request.data.get('start')
            end_location = request.data.get('end')

            if not start_location or not end_location:
                return Response(
                    {'error': 'Start and end locations are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            calculator = RouteCalculator()
            
            # Log the process
            logger.info(f"Calculating route from {start_location} to {end_location}")
            
            # Get coordinates
            start_coords = calculator.get_coordinates(start_location)
            end_coords = calculator.get_coordinates(end_location)
            
            # Calculate route and fuel stops
            fuel_stops, total_cost = calculator.find_optimal_fuel_stops(start_coords, end_coords)
            
            # Get route map
            route_map = calculator.get_route_map(start_coords, end_coords, fuel_stops)
            
            response_data = {
                'route_map': route_map,
                'fuel_stops': fuel_stops,
                'total_fuel_cost': round(total_cost, 2),
                'start_coords': start_coords,
                'end_coords': end_coords
            }
            
            return Response(response_data)
            
        except Exception as e:
            logger.error(f"Error processing route: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )