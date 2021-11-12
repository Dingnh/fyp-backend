from PIL.Image import Image
from django.shortcuts import render

from django.contrib.auth.models import User, Group
from facemask.quickstart.serializers import UserSerializer, GroupSerializer
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.http import HttpResponse
from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from rest_framework.decorators import action
from django.views.decorators.csrf import csrf_exempt
from .serviceclass import ImageClassificationModel
from .serializers import FaceMaskSerializer, FaceMaskCreate
from .models import FaceMaskModel


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]


class FaceMaskViewSet(viewsets.ModelViewSet):
    serializer_class = FaceMaskSerializer
    model = FaceMaskModel
    queryset = FaceMaskModel.objects.all()

    @action(detail=True, methods=['post'])
    def upload_docs(request):
        try:
            file = request.data['file']
            filesuploaded = ["with black mask.png",
                             "noisy_masked_person.png", "with blue mask.jpg"]

            output_images = ImageClassificationModel.processImagesInput(
                file)
            # TODO: Implment filter here
            return output_images
        except KeyError:
            raise ('Request has no resource file attached')


@csrf_exempt
def detect(request):
    data = {"success": False}
    if request.method == "POST":
        if request.FILES is not None:
            output_image = ImageClassificationModel.processImagesInput(
                request.FILES['image'])
        else:
            data["error"] = "No IMAGES provided."
            return JsonResponse(data)
        output_image = str(output_image, 'ascii')
        data.update({"output_image": output_image, "success": True})
    return JsonResponse(data)
