from django.db import models
from django.db.models.fields import SlugField
from django.utils import timezone
from django.contrib.auth.models import User

# Create your models here.


class FaceMaskModel(models.Model):
    title = models.CharField(max_length=1024)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
