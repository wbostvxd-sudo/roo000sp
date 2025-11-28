import os
import sys
import shutil
import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predictor import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path
import gradio as gr

# Initialize globals
roop.globals.source_path = None
roop.globals.target_path = None
roop.globals.output_path = None
roop.globals.headless = True
roop.globals.frame_processors = ['face_swapper']
roop.globals.keep_fps = True
roop.globals.keep_frames = False
roop.globals.skip_audio = False
roop.globals.many_faces = False
roop.globals.reference_face_position = 0
roop.globals.reference_frame_number = 0
roop.globals.similar_face_distance = 0.85
roop.globals.temp_frame_format = 'png'
roop.globals.temp_frame_quality = 100
roop.globals.output_video_encoder = 'libx264'
roop.globals.output_video_quality = 35
roop.globals.max_memory = 60
roop.globals.execution_providers = ['CUDAExecutionProvider']
roop.globals.execution_threads = 8

def process_media(
    source_img,
    target_path,
    frame_processors,
    keep_fps,
    skip_audio,
    many_faces,
    reference_face_position,
    similar_face_distance,
    temp_frame_format,
    temp_frame_quality,
    output_video_encoder,
    output_video_quality,
    max_memory,
    execution_threads
):
    if not source_img or not target_path:
        return None
    
    # Set globals
    roop.globals.source_path = source_img
    roop.globals.target_path = target_path
    roop.globals.frame_processors = frame_processors
    roop.globals.keep_fps = keep_fps
    roop.globals.skip_audio = skip_audio
    roop.globals.many_faces = many_faces
    roop.globals.reference_face_position = reference_face_position
    roop.globals.similar_face_distance = similar_face_distance
    roop.globals.temp_frame_format = temp_frame_format
    roop.globals.temp_frame_quality = temp_frame_quality
    roop.globals.output_video_encoder = output_video_encoder
    roop.globals.output_video_quality = output_video_quality
    roop.globals.max_memory = max_memory
    roop.globals.execution_threads = execution_threads
    
    # Generate output path
    output_dir = os.path.dirname(target_path)
    output_filename = f"swapped_{os.path.basename(target_path)}"
    roop.globals.output_path = os.path.join(output_dir, output_filename)
    
    # Start processing (logic from roop.core.start)
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return "Error in pre_check"
        if not frame_processor.pre_start():
            return "Error in pre_start"

    # process image to image
    if has_image_extension(roop.globals.target_path):
        if predict_image(roop.globals.target_path):
            return "NSFW detected"
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        # process frame
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
            frame_processor.post_process()
        return roop.globals.output_path

    # process image to videos
    if predict_video(roop.globals.target_path):
        return "NSFW detected"
        
    create_temp(roop.globals.target_path)
    
    # extract frames
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        extract_frames(roop.globals.target_path, fps)
    else:
        extract_frames(roop.globals.target_path)
    
    # process frame
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        return "Frames not found"
        
    # create video
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        create_video(roop.globals.target_path, fps)
    else:
        create_video(roop.globals.target_path)
    
    # handle audio
    if roop.globals.skip_audio:
        move_temp(roop.globals.target_path, roop.globals.output_path)
    else:
        restore_audio(roop.globals.target_path, roop.globals.output_path)
        
    # clean temp
    clean_temp(roop.globals.target_path)
    
    return roop.globals.output_path

# Gradio Interface
with gr.Blocks(title="Roop Face Swap") as app:
    gr.Markdown("# Roop Face Swap (Colab Version)")
    
    with gr.Row():
        with gr.Column():
            source_image = gr.Image(type="filepath", label="Source Face")
            target_media = gr.File(label="Target Image/Video")
            submit_btn = gr.Button("Start Face Swap", variant="primary")
            
        with gr.Column():
            output_media = gr.File(label="Output")
            
    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            frame_processors = gr.CheckboxGroup(
                choices=['face_swapper', 'face_enhancer'],
                value=['face_swapper'],
                label="Frame Processors"
            )
            many_faces = gr.Checkbox(label="Many Faces", value=False)
            skip_audio = gr.Checkbox(label="Skip Audio", value=False)
            keep_fps = gr.Checkbox(label="Keep FPS", value=True)
            
        with gr.Row():
            similar_face_distance = gr.Slider(
                minimum=0.0, maximum=1.5, value=0.85, step=0.01, label="Similar Face Distance"
            )
            reference_face_position = gr.Number(value=0, label="Reference Face Position", precision=0)
            
        with gr.Row():
            output_video_encoder = gr.Dropdown(
                choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'],
                value='libx264',
                label="Output Video Encoder"
            )
            output_video_quality = gr.Slider(
                minimum=0, maximum=100, value=35, step=1, label="Output Video Quality"
            )
            
        with gr.Row():
            temp_frame_format = gr.Dropdown(
                choices=['png', 'jpg'], value='png', label="Temp Frame Format"
            )
            temp_frame_quality = gr.Slider(
                minimum=0, maximum=100, value=100, step=1, label="Temp Frame Quality"
            )
            
        with gr.Row():
            max_memory = gr.Number(value=60, label="Max Memory (GB)")
            execution_threads = gr.Number(value=8, label="Execution Threads", precision=0)

    submit_btn.click(
        fn=process_media,
        inputs=[
            source_image,
            target_media,
            frame_processors,
            keep_fps,
            skip_audio,
            many_faces,
            reference_face_position,
            similar_face_distance,
            temp_frame_format,
            temp_frame_quality,
            output_video_encoder,
            output_video_quality,
            max_memory,
            execution_threads
        ],
        outputs=[output_media]
    )

# Check if running on Colab
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

if __name__ == "__main__":
    app.launch(share=IS_COLAB)
