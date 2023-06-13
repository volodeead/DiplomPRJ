# currency_app/views.py
from datetime import datetime, timedelta
import requests
from django.http import JsonResponse
from django.views.generic import TemplateView
from .models import ExchangeRate
from datetime import datetime, timedelta
from keras.models import load_model
import pickle
import numpy as np
import os
from django.conf import settings
import yfinance as yf

from statsmodels.tsa.arima.model import ARIMAResults

# Перетворіть дані для LSTM
def create_dataset(dataset, look_back=1):
    X = []
    for i in range(len(dataset)-look_back+1):
        a = dataset[i:(i+look_back),0]
        X.append(a)
    return np.array(X)

# Перетворіть дані для GRU
def create_dataset2(dataset, look_back=1):
    dataX = []
    for i in range(look_back, len(dataset)):
        a = dataset[i-look_back:i, 0]
        dataX.append(a)
    return np.array(dataX)


modelLSTM5min_path = os.path.join(settings.BASE_DIR, 'currency_app', 'Model_LSTM_Solo.h5')
scalerLSTM5min_path = os.path.join(settings.BASE_DIR, 'currency_app', 'scalerLSTM5min.pkl')

modelGRU5min_path = os.path.join(settings.BASE_DIR, 'currency_app', 'GRU5min.h5')
scaler_path = os.path.join(settings.BASE_DIR, 'currency_app', 'scalerGRU.pkl')

modelGRU5min = load_model(modelGRU5min_path)
modelLSTM5min = load_model(modelLSTM5min_path)

# Load the scaler
with open(scaler_path, 'rb') as f:
    scalerGRU5min = pickle.load(f)

# Завантажте збережений скейлер
with open(scalerLSTM5min_path, 'rb') as f:
    scalerLSTM5min = pickle.load(f)



class ExchangeRateChartView(TemplateView):
    template_name = 'currency_chart.html'


    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['qs'] = ExchangeRate.objects.all().order_by('timestamp')
        return context

    def fetch_exchange_rate(self):

        url = "https://api.apilayer.com/exchangerates_data/latest?symbols=usd&base=eur"
        payload = {}
        headers= {
        "apikey": "pRtV3KddGPwMSoSfTNSeP4aM5nkIdYjH"
        }

        BASE_CURRENCY = 'EUR'
        TARGET_CURRENCY = 'USD'

        response = requests.request("GET", url, headers=headers, data = payload)

        print(response.status_code)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                exchange_rate = data["rates"].get("USD")
                if exchange_rate is not None:
                    timestamp = datetime.fromtimestamp(data["timestamp"])
                    ExchangeRate.objects.create(
                        base_currency=BASE_CURRENCY,
                        target_currency=TARGET_CURRENCY,
                        exchange_rate=exchange_rate,
                        timestamp=timestamp
                    )

def latest_exchange_rate(request):
    latest_rate = ExchangeRate.objects.latest('timestamp')
    data = {
        'timestamp': latest_rate.timestamp_milliseconds,
        'exchange_rate': latest_rate.exchange_rate
    }
    return JsonResponse(data)


def predict5minLSTM(request):
    look_back = 7
    # Get last 7 values. You need to replace this with your own data fetching logic
    last_seven_values = list(ExchangeRate.objects.order_by('-timestamp').values_list('exchange_rate', flat=True)[:7][::-1])
    print(last_seven_values)

    # Normalize data
    last_seven_values = scalerLSTM5min.transform(np.array(last_seven_values).reshape(-1, 1))
    dataX = create_dataset(last_seven_values, look_back)
    print(dataX)
    dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))


    # Predict next value
    next_value = modelLSTM5min.predict(dataX)
    
    # De-normalize predicted value
    next_value = scalerLSTM5min.inverse_transform(next_value)
    
    return JsonResponse({'next_5min_value': float(next_value)})

def predict10minLSTM(request):
    look_back = 7
    steps = 2  # Predict two steps ahead

    # Get last 7 values. You need to replace this with your own data fetching logic
    last_seven_values = list(ExchangeRate.objects.order_by('-timestamp').values_list('exchange_rate', flat=True)[:7][::-1])

    # Get the latest timestamp
    latest_timestamp = ExchangeRate.objects.order_by('-timestamp').first().timestamp

    # Normalize data
    last_seven_values = scalerLSTM5min.transform(np.array(last_seven_values).reshape(-1, 1))
    dataX = create_dataset(last_seven_values, look_back)
    dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))

    predictions = []
    for i in range(steps):
        # Predict next value
        next_value = modelLSTM5min.predict(dataX)

        # Append prediction to the list with corresponding timestamp
        next_timestamp = latest_timestamp + timedelta(minutes=5*(i+1))
        predictions.append({"timestamp": next_timestamp.isoformat(), "exchange_rate": float(scalerLSTM5min.inverse_transform(next_value)[0][0])})

        # Add new prediction to the end of dataX and remove the first element
        dataX = np.append(dataX[:, 1:, :], next_value.reshape(1, 1, 1), axis=1)

    return JsonResponse({'next_10min_value': predictions})

def predict30minLSTM(request):
    look_back = 7
    steps = 6  # Predict 6 steps ahead

    # Get last 7 values. You need to replace this with your own data fetching logic
    last_seven_values = list(ExchangeRate.objects.order_by('-timestamp').values_list('exchange_rate', flat=True)[:7][::-1])

    # Get the latest timestamp
    latest_timestamp = ExchangeRate.objects.order_by('-timestamp').first().timestamp

    # Normalize data
    last_seven_values = scalerLSTM5min.transform(np.array(last_seven_values).reshape(-1, 1))
    dataX = create_dataset(last_seven_values, look_back)
    dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))

    predictions = []
    for i in range(steps):
        # Predict next value
        next_value = modelLSTM5min.predict(dataX)

        # Append prediction to the list with corresponding timestamp
        next_timestamp = latest_timestamp + timedelta(minutes=5*(i+1))
        predictions.append({"timestamp": next_timestamp.isoformat(), "exchange_rate": float(scalerLSTM5min.inverse_transform(next_value)[0][0])})

        # Add new prediction to the end of dataX and remove the first element
        dataX = np.append(dataX[:, 1:, :], next_value.reshape(1, 1, 1), axis=1)

    return JsonResponse({'next_30min_value': predictions})


def predict5minGRU(request):
    look_back = 7

    # Get last 7 values
    last_seven_values = list(ExchangeRate.objects.order_by('-timestamp').values_list('exchange_rate', flat=True)[:7][::-1])

    # Normalize data
    last_seven_values = scalerGRU5min.transform(np.array(last_seven_values).reshape(-1, 1))

    # Reshape data to be [samples, timesteps, features]
    dataX = last_seven_values.reshape(1, look_back, 1)

    # Predict next value
    next_value = modelGRU5min.predict(dataX)

    # De-normalize predicted value
    next_value = scalerGRU5min.inverse_transform(next_value)

    return JsonResponse({'next_5min_value': float(next_value[0, 0])})

def predict10minGRU(request):
    look_back = 7
    steps = 2

    last_seven_values = list(ExchangeRate.objects.order_by('-timestamp').values_list('exchange_rate', flat=True)[:7][::-1])
    latest_timestamp = ExchangeRate.objects.order_by('-timestamp').first().timestamp

    last_seven_values = scalerGRU5min.transform(np.array(last_seven_values).reshape(-1, 1))

    dataX = last_seven_values.reshape(1, look_back, 1)

    predictions = []
    for i in range(steps):
        next_value = modelGRU5min.predict(dataX)
        next_timestamp = latest_timestamp + timedelta(minutes=5 * (i + 1))
        predictions.append({"timestamp": next_timestamp.isoformat(), "exchange_rate": float(scalerGRU5min.inverse_transform(next_value)[0][0])})
        dataX = np.append(dataX[:, 1:, :], next_value.reshape(1, 1, 1), axis=1)

    return JsonResponse({'next_10min_value': predictions})

def predict30minGRU(request):
    look_back = 7
    steps = 6

    last_seven_values = list(ExchangeRate.objects.order_by('-timestamp').values_list('exchange_rate', flat=True)[:7][::-1])
    latest_timestamp = ExchangeRate.objects.order_by('-timestamp').first().timestamp

    last_seven_values = scalerGRU5min.transform(np.array(last_seven_values).reshape(-1, 1))

    dataX = last_seven_values.reshape(1, look_back, 1)

    predictions = []
    for i in range(steps):
        next_value = modelGRU5min.predict(dataX)
        next_timestamp = latest_timestamp + timedelta(minutes=5 * (i + 1))
        predictions.append({"timestamp": next_timestamp.isoformat(), "exchange_rate": float(scalerGRU5min.inverse_transform(next_value)[0][0])})
        dataX = np.append(dataX[:, 1:, :], next_value.reshape(1, 1, 1), axis=1)

    return JsonResponse({'next_30min_value': predictions})


def predict1monthARIMA(request):

    arima1month = os.path.join(settings.BASE_DIR, 'currency_app', 'model_arima.pkl')
    loaded_model = ARIMAResults.load(arima1month)

    last_day_index = len(dataYahooFinance1year)  # індекс останнього дня даних
    last_day_date = dataYahooFinance1year.index[-1]  # timestamp останнього дня даних

    # Прогноз на 30 днів від останнього значення в data
    forecast = loaded_model.predict(start=last_day_index, end=last_day_index + 30)

    # Створення списку пар [timestamp, rate]
    forecast_list = []
    for i in range(30):
        timestamp = (last_day_date + timedelta(days=i+1)).timestamp()
        rate = forecast[i+1]
        forecast_list.append([timestamp, rate])


    return JsonResponse({'next_month_value': forecast_list})

def predict3daysARIMA(request):

    arima1month = os.path.join(settings.BASE_DIR, 'currency_app', 'model_arima.pkl')
    loaded_model = ARIMAResults.load(arima1month)

    last_day_index = len(dataYahooFinance1year)  # індекс останнього дня даних
    last_day_date = dataYahooFinance1year.index[-1]  # timestamp останнього дня даних

    # Прогноз на 3 дні від останнього значення в data
    forecast = loaded_model.predict(start=last_day_index, end=last_day_index + 3)

    # Створення списку пар [timestamp, rate]
    forecast_list = []
    for i in range(3):
        timestamp = (last_day_date + timedelta(days=i+1)).timestamp()
        rate = forecast[i+1]
        forecast_list.append([timestamp, rate])


    return JsonResponse({'next_month_value': forecast_list})

def predict1weekARIMA(request):

    arima1month = os.path.join(settings.BASE_DIR, 'currency_app', 'model_arima.pkl')
    loaded_model = ARIMAResults.load(arima1month)

    last_day_index = len(dataYahooFinance1year)  # індекс останнього дня даних
    last_day_date = dataYahooFinance1year.index[-1]  # timestamp останнього дня даних

    # Прогноз на 3 дні від останнього значення в data
    forecast = loaded_model.predict(start=last_day_index, end=last_day_index + 7)

    # Створення списку пар [timestamp, rate]
    forecast_list = []
    for i in range(7):
        timestamp = (last_day_date + timedelta(days=i+1)).timestamp()
        rate = forecast[i+1]
        forecast_list.append([timestamp, rate])


    return JsonResponse({'next_month_value': forecast_list})

def predict3monthsARIMA(request):

    arima1month = os.path.join(settings.BASE_DIR, 'currency_app', 'model_arima.pkl')
    loaded_model = ARIMAResults.load(arima1month)

    last_day_index = len(dataYahooFinance1year)  # індекс останнього дня даних
    last_day_date = dataYahooFinance1year.index[-1]  # timestamp останнього дня даних

    # Прогноз на 3 дні від останнього значення в data
    forecast = loaded_model.predict(start=last_day_index, end=last_day_index + 90)

    # Створення списку пар [timestamp, rate]
    forecast_list = []
    for i in range(90):
        timestamp = (last_day_date + timedelta(days=i+1)).timestamp()
        rate = forecast[i+1]
        forecast_list.append([timestamp, rate])


    return JsonResponse({'next_month_value': forecast_list})


def predict3daysProphet(request):

    prophetPATH = os.path.join(settings.BASE_DIR, 'currency_app', 'model_prophet.pkl')
    with open(prophetPATH, 'rb') as f:
        loaded_prophet = pickle.load(f)

    future = loaded_prophet.make_future_dataframe(periods=3)
    # Generate forecast
    forecast = loaded_prophet.predict(future)

    # Convert DataFrame to list of dictionaries
    forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3).to_dict('records')

    return JsonResponse({'next_3days_value': forecast_dict})

def predict1weekProphet(request):

    prophetPATH = os.path.join(settings.BASE_DIR, 'currency_app', 'model_prophet.pkl')
    with open(prophetPATH, 'rb') as f:
        loaded_prophet = pickle.load(f)

    future = loaded_prophet.make_future_dataframe(periods=7)
    # Generate forecast
    forecast = loaded_prophet.predict(future)

    # Convert DataFrame to list of dictionaries
    forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).to_dict('records')

    return JsonResponse({'next_1week_value': forecast_dict})

def predict1monthProphet(request):

    prophetPATH = os.path.join(settings.BASE_DIR, 'currency_app', 'model_prophet.pkl')
    with open(prophetPATH, 'rb') as f:
        loaded_prophet = pickle.load(f)

    future = loaded_prophet.make_future_dataframe(periods=30)
    # Generate forecast
    forecast = loaded_prophet.predict(future)

    # Convert DataFrame to list of dictionaries
    forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict('records')

    return JsonResponse({'next_1month_value': forecast_dict})

def predict3monthsProphet(request):

    prophetPATH = os.path.join(settings.BASE_DIR, 'currency_app', 'model_prophet.pkl')
    with open(prophetPATH, 'rb') as f:
        loaded_prophet = pickle.load(f)

    future = loaded_prophet.make_future_dataframe(periods=90)
    # Generate forecast
    forecast = loaded_prophet.predict(future)

    # Convert DataFrame to list of dictionaries
    forecast_dict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(90).to_dict('records')

    return JsonResponse({'next_3months_value': forecast_dict})

dataYahooFinance1year = 0
# in the view that handles /latest_rate or /predict
def yfinance1year(request):
    global dataYahooFinance1year # declare the variable as global
    data = yf.download('EURUSD=X', period='366d', interval='1d')
    data = data['Close'].dropna()
    dataYahooFinance1year = data

    json_data = data.to_json()

    return JsonResponse(json_data, safe=False)