# Generated by Django 3.1.7 on 2021-04-22 05:10

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='danger',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('solver', models.CharField(max_length=10)),
                ('area_name', models.CharField(max_length=10)),
            ],
            options={
                'verbose_name': '危险情况表',
                'db_table': 'danger',
            },
        ),
    ]