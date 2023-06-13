# currency_app/models.py
from django.db import models

class ExchangeRate(models.Model):
    base_currency = models.CharField(max_length=3)
    target_currency = models.CharField(max_length=3)
    exchange_rate = models.FloatField()
    timestamp = models.DateTimeField()

    @property
    def timestamp_milliseconds(self):
        return int(self.timestamp.timestamp() * 1000)