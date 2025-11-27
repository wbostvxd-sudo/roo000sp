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

def process_media(source_img, target_path, face_enhancer):
    if not source_img or not target_path:
        return None
    
    # Set globals
    roop.globals.source_path = source_img
    roop.globals.target_path = target_path
    
    # Configure processors
    processors = ['face_swapper']
    if face_enhancer:
        processors.append('face_enhancer')
    roop.globals.frame_processors = processors
    
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
    fps = detect_fps(roop.globals.target_path)
    extract_frames(roop.globals.target_path, fps)
    
    # process frame
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        return "Frames not found"
        
    # create video
    create_video(roop.globals.target_path, fps)
    
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
            face_enhancer = gr.Checkbox(label="Face Enhancer (GFPGAN)", value=False)
            submit_btn = gr.Button("Start Face Swap")
        
        with gr.Column():
            output_media = gr.File(label="Output")
    
    submit_btn.click(
        fn=process_media,
        inputs=[source_image, target_media, face_enhancer],
        outputs=[output_media]
    )

if __name__ == "__main__":
    app.launch(share=True)
