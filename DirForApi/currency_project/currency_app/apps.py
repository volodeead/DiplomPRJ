# currency_app/apps.py
from django.apps import AppConfig
from apscheduler.schedulers.background import BackgroundScheduler

def startPRG():
        schedule = BackgroundScheduler()
        from .views import ExchangeRateChartView
        currency = ExchangeRateChartView()
        schedule.add_job(currency.fetch_exchange_rate, "interval", minutes=5, id="currency_update_001", replace_existing=True)
        schedule.start()

class CurrencyAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'currency_app'

    def ready(self):
        print("Starting Schedule")
        startPRG()
