from django.db import models

class FuelStation(models.Model):
    location = models.CharField(max_length=255)
    price = models.FloatField()
    latitude = models.FloatField()
    longitude = models.FloatField()

    def __str__(self):
        return f"{self.location} - ${self.price}"