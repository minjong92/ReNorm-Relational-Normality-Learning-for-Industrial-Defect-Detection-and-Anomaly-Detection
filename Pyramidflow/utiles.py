import time
from datetime import datetime, timedelta


def remaining_time(since, epoch, epochs, start_epoch=0):
	
    time_now = time.time()
    elapsed = time_now - since
    epochs_done = (epoch - start_epoch + 1)
    avg_epoch_time = elapsed / epochs_done if epochs_done > 0 else 0
    epochs_left = (epochs - 1) - epoch
    eta_sec = avg_epoch_time * epochs_left

    return str(timedelta(seconds=int(eta_sec)))