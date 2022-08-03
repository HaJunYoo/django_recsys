from django.db import models

class PredResults(models.Model):
    name = models.CharField(max_length=80)
    img = models.CharField(max_length=80)
    review = models.TextField(max_length=256, default='default')
    price = models.CharField(max_length=80, default='default')

    def __str__(self):
        return f'{self.name}'