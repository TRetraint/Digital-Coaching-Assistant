for %%f in (.\pull_up_data\videos\*) do (
python run_webcam.py --camera %%f --model cmu
)
PAUSE