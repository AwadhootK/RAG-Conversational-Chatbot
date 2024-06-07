import os
import shutil
import boto3


def check_and_clean_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def download_files_from_s3(username):
    s3 = boto3.resource('s3')
    bucket_name = 'awadhoot-rag-chatbot-files'
    bucket = s3.Bucket(bucket_name)
    s3_folder = username
    local_dir = None

    check_and_clean_folder("docs")

    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

    os.rename(s3_folder, "docs")


download_files_from_s3("awadhootk7")
