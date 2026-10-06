from pytubefix import YouTube
from .mongo_repo import MongoRepository


def download_videoid(video_id: str, true_label: str, out_path: str = r"G:\UNI\TFG\Videos"):
    url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(url)


    #stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()   #parametros concretos
    stream = yt.streams.get_highest_resolution()
    file_path = stream.download(output_path=out_path, filename=f"{video_id}.mp4")

    video_doc = {
        "video_id": yt.video_id,
        "title": yt.title,
        "views": yt.views,
        "length": yt.length,
        "author": yt.author,
        "file_path": file_path,
        "true_label": true_label
    }

    repo = MongoRepository()
    repo.insert_one(video_doc)

    return video_doc


def download_from_url(url: str, true_label: str = None, out_path: str = r"G:\UNI\TFG\Videos"):
    video_doc = None
    repo = MongoRepository()
    try:
        yt = YouTube(url)

        #stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()   #parametros concretos
        stream = yt.streams.get_highest_resolution()
        file_path = stream.download(output_path=out_path, filename=f"{yt.video_id}.mp4")

        video_doc = {
            "video_id": yt.video_id,
            "title": yt.title,
            "views": yt.views,
            "length": yt.length,
            "author": yt.author,
            "file_path": file_path,
            "true_label": true_label
        }
        
        repo.insert_one(video_doc)
    
    except Exception as e:
        print(f"Error descargando video: {e}")
    
    return video_doc