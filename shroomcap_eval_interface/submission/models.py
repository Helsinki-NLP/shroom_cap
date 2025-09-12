from django.db import models
from django.contrib.auth import models as auth_models
from django.utils.translation import gettext_lazy as _


class Profile(models.Model):
    participant = models.OneToOneField(auth_models.User, on_delete=models.CASCADE)
    team_name = models.CharField(max_length=200)


class Language(models.TextChoices):
    EN = 'EN', _('English')
    ES = 'ES', _('Spanish')
    FR = 'FR', _('French')
    HI = 'HI', _('Hindi')
    IT = 'IT', _('Italian')
    TE = 'TE', _('Telugu')
    GU = 'GU', _('Gujarati')
    ML = 'ML', _('Malayalam')


class DataSplit(models.TextChoices):
    val = 'VAL', _('Val')
    test = 'TST', _('Test')


class Submission(models.Model):
    identifier = models.CharField(max_length=50)
    system_description = models.TextField()
    is_prompt = models.BooleanField()
    is_rag = models.BooleanField()
    dataset_description = models.TextField()
    plms_description = models.TextField()
    extra_description = models.TextField()
    submitter = models.ForeignKey(Profile, on_delete=models.CASCADE)
    language = models.CharField(max_length=2, choices=Language.choices)
    split = models.CharField(max_length=3, choices=DataSplit.choices)
    fact_score = models.FloatField()
    flue_score = models.FloatField()


class DataPoint(models.Model):
    submission = models.ForeignKey(Submission, on_delete=models.CASCADE, blank=True, null=True)
    datapoint_id = models.CharField(max_length=20)
    split = models.CharField(max_length=3, choices=DataSplit.choices)
    language = models.CharField(max_length=2, choices=Language.choices)
    factual = models.BooleanField()
    fluent = models.BooleanField()
    is_ref = models.BooleanField(default=False)

