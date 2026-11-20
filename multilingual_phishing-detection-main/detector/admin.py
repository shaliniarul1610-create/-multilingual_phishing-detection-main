from django.contrib import admin
from .models import ScanHistory

@admin.register(ScanHistory)
class ScanHistoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'created_at', 'language', 'result', 'verdict', 'threat_score', 'reported_false_positive')
    list_filter = ('language', 'result', 'verdict', 'reported_false_positive')
    list_editable = ('reported_false_positive',)
    search_fields = ('sender_domain', 'email_text', 'user__username')