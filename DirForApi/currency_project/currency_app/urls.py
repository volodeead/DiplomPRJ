# currency_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('predict5minLSTM/', views.predict5minLSTM, name='predict5minLSTM'),
    path('predict10minLSTM/', views.predict10minLSTM, name='predict10minLSTM'),
    path('predict30minLSTM/', views.predict30minLSTM, name='predict30minLSTM'),

    path('predict5minGRU/', views.predict5minGRU, name='predict5minGRU'),
    path('predict10minGRU/', views.predict10minGRU, name='predict10minGRU'),
    path('predict30minGRU/', views.predict30minGRU, name='predict30minGRU'),

    path('predict1monthARIMA/', views.predict1monthARIMA, name='predict1monthARIMA'),
    path('predict3monthsARIMA/', views.predict3monthsARIMA, name='predict3monthsARIMA'),
    path('predict3daysARIMA/', views.predict3daysARIMA, name='predict3daysARIMA'),
    path('predict1weekARIMA/', views.predict1weekARIMA, name='predict1weekARIMA'),

    path('predict3daysProphet/', views.predict3daysProphet, name='predict3daysProphet'),
    path('predict1weekProphet/', views.predict1weekProphet, name='predict1weekProphet'),
    path('predict1monthProphet/', views.predict1monthProphet, name='predict1monthProphet'),
    path('predict3monthsProphet/', views.predict3monthsProphet, name='predict3monthsProphet'),

    path('currency_chart/', views.ExchangeRateChartView.as_view(), name='currency_chart'),
    path('latest_rate/', views.latest_exchange_rate, name='latest_rate'),  # Новий URL
    path('yfinance1year/', views.yfinance1year, name='yfinance1year'),  # Новий URL

]
