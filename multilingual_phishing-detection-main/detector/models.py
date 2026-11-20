from django.conf import settings
from django.db import models

class ScanHistory(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="scans"
    )
    email_text = models.TextField()
    language = models.CharField(max_length=50, blank=True, null=True)
    sender_domain = models.CharField(max_length=255, blank=True, null=True)
    ml_score = models.FloatField(default=0.0)
    threat_score = models.FloatField(default=0.0)
    result = models.CharField(max_length=20, default="Unknown")
    verdict = models.CharField(max_length=50)
    flags = models.TextField(blank=True, null=True)
    reported_false_positive = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.verdict} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"