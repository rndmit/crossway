import os

class AppConfig():
    tweak_mode = True
    tm_threshold_step = 0.01
    stream_url = 'https://sochi.camera:8081/cam_274/tracks-v1/mono.m3u8'
    #stream_url = os.path.join(os.path.abspath(os.path.dirname(__file__)), "input.mkv")
    mqtt_username = "13fa60a0f3e5d2416b4ede1ea38a33684117f57325b38bc57476bc4b7c2c"

class NNConfig():
    human_threshold = 0.2
    car_threshold = 0.4
    obj="car"
    run=obj+""
    percentage=0.6
    sample_dir="./crossway/sample_data"
    dataset_dir=sample_dir+"/dataset"
    image_dir=sample_dir + "/training"
    train_image_dir="./crossway/sample_data/training"
    test_image_dir="./crossway/sample_data/test"
    ckpt_dir = "./crossway/ckpts/"+run
    batch_size = 4
    width=1280
    height=720
    patch_width_height=256
    base_filter_size=16
    learning_rate=0.001
    pos_weight=5.0
    steps=100
    secs=1000
    train_steps=100
    use_skip_connections=False