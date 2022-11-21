from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from IPython.display import Video

def create_video(row, folder_name,before=5, after=5):
    print(row["event_attributes"])
    filename = f"./videos/{folder_name}/test_{row['index']}.mp4"
    ffmpeg_extract_subclip(
        f"./videos/train/{row['video_id']}.mp4", 
        int(row['time']) - before, 
        int(row['time']) + after, 
        targetname=filename,
    )
    
    return Video(filename, width=800),str(row["event_attributes"]),filename
