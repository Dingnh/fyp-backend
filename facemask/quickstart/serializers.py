from django.contrib.auth.models import User, Group
from django.db.models import fields
from rest_framework import serializers
from .models import FaceMaskModel


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ['url', 'username', 'email', 'groups']


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ['url', 'name']


class FaceMaskCreate(serializers.Serializer):
    title = serializers.CharField(max_length=1024)
    description = serializers.CharField()


class FaceMaskSerializer(serializers.ModelSerializer):

    class Meta:
        model = FaceMaskModel
        fields = '__all__'
