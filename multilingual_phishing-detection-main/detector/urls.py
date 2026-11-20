from django.urls import path
from . import views

urlpatterns = [
    # landing + auth
    path('', views.home, name='home'),
    path('login/', views.user_login, name='user_login'),
    path('signup/', views.signup_view, name='signup'),
    path('admin-login/', views.admin_login, name='admin_login'),
    path('logout/', views.logout_view, name='logout'),

    # user area
    path('scanner/', views.predict_email, name='predict_email'),
    path('history/', views.scan_history, name='scan_history'),
    path('analytics/', views.threat_analytics, name='threat_analytics'),
    path('report/<int:scan_id>/false-positive/', views.report_false_positive, name='report_false_positive'),

    # shared reports (user: own scans, admin: any scan)
    path('report/<int:scan_id>/', views.forensic_report, name='forensic_report'),
    path('report/<int:scan_id>/pdf/', views.forensic_report_pdf, name='forensic_report_pdf'),

    # admin area
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-history/', views.admin_history, name='admin_history'),
    path('admin-analytics/', views.admin_analytics, name='admin_analytics'),
]