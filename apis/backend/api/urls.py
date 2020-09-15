from django.urls import path

from .views import *

urlpatterns = [
    path('vehicle', VehicleDetection.as_view(), name="vehicle detection api"),
    path('lpinvehicle', LicensePlateInVehicleDetection.as_view(), name="license plate detection in vehicle api"),
    path('lp', LicensePlateDetection.as_view(), name="license plate detection api"),
]
