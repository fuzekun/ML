# Generated by Django 3.1.7 on 2021-04-25 04:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('firstWEB', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='areas',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('areas_name', models.CharField(max_length=10)),
                ('remarks', models.CharField(max_length=10)),
                ('director', models.CharField(max_length=10)),
            ],
        ),
        migrations.CreateModel(
            name='camera',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Ip', models.CharField(max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name='danger',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('solver', models.CharField(max_length=10)),
                ('area_name', models.CharField(max_length=10)),
            ],
        ),
        migrations.CreateModel(
            name='vedio',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('img_id', models.CharField(max_length=20)),
                ('img_url', models.CharField(max_length=20)),
            ],
        ),
    ]