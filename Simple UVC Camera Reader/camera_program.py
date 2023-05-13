"""
Capture images from UVC using OpenCV in a way that spreads the load between processor cores for full speed.
Processes are arranged like this:
________________             ________________            ________________
| camera_worker | --queue--> | logic_worker | --queue--> | Main Process |
-----------------            ----------------            ----------------
One process's output queue is the next's input queue

"""
import multiprocessing
import os
import time
import numpy as np
import cv2

DISPLAY_HEIGHT = 1242
DISPLAY_WIDTH = 2208
USE_DEPTH = False

def cam_proc_cv2(exit_event, output_queue):
    """ A worker process to capture images from the camera. Output images are put into the output queue. """
    import cv2

    # Open the ZED camera using UVC
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == 0:
        output_queue.put(None)
        return

    # Set the camera resolution. See camera docs for resolutions.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560) # Note zed2 is 2 images wide.
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while not exit_event.is_set() : # Exit event kills the process
        # Get a new frame from camera
        retval, frame = cap.read()

        # Extract left and right images from side-by-side
        left_right_image = np.split(frame, 2, axis=1)
        image = left_right_image[0] # [0] is left, [1] is right

        # To crop the image do this:
        # image = image[ROW_START:ROW_END,COL_START:COL_END,:3]

        # Shove the image into the queue so the next process can read it
        # Stop putting in images if the next process isn't keeping up
        # Add more checks for other queues here if needed
        if output_queue.qsize() < 1:
            output_queue.put(image)
    cap.release()
    exit()


def logic_proc(exit_event, input_queue, output_queue):
    """ A worker process to evaluate the image and make a decision about it."""
    try:
        while not exit_event.is_set(): #Exit event kills the process
            image = input_queue.get(block=True, timeout=2)
            if image is None:
                output_queue.put(None)
                break

            logic_result = True # Define custom logic here. Doesn't have to be a boolean

            # Any code that slows down the framerate should go here
            # or, if in another process, we should add an input queue length check
            # for that process's queue in the camera process (don't add images if we aren't keeping up!)

            # Pack the image along with the logic_result so that the next
            # process has both
            message = (logic_result, image)
            output_queue.put(message)
    except multiprocessing.queues.Empty:
        pass
    except KeyboardInterrupt:
        pass


if __name__ == '__main__': # If this file was run directly run everything below.
    exit_event = multiprocessing.Event()
    image_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    # More queues could go here if needed for more workers

    # Worker process to read images from the camera
    camera_worker = multiprocessing.Process(name='camera_worker', target=cam_proc_cv2, args=(exit_event, image_queue))
    camera_worker.start()

    # Worker process to evaluate the image and make a decision about it
    logic_worker = multiprocessing.Process(name='logic_worker', target=logic_proc, args=(exit_event, image_queue, result_queue))
    logic_worker.start()

    # More workers could go here: maybe one for commuications with the plc

    count = -1
    try:
        while True:  #Main loop
            #Get the current time in seconds since epoch start in 1970
            start_time = time.time()
            message = result_queue.get() # Blocks until something is available
            if message is None:
                break
            logic_result, image = message # unpack the message from the queue

            # Display the image in a window called "Camera View"
            cv2.imshow("Camera View", image)
            key = cv2.waitKey(1)
            if key == 27: # ESC pressed
                break

            # Print the loop frame rate in frames per second
            print(1.0 / (time.time() - start_time))
    except KeyboardInterrupt:
        print('Exiting')
    finally:
        exit_event.set() #Attached to the camera_worker process. Will trigger an exit message to worker processes
        cv2.destroyAllWindows()

        # Clean up camera processes.
        camera_worker.join(2) # Wait 1 seconds for logic_worker to gracefully terminate
        if camera_worker.exitcode is None:
            # the camera_worker subprocess did not terminate, try to terminate it forcefully
            print("Subprocess 'camera_worker' did not terminate properly, sending SIGTERM...")
            print("Not a big deal, just odd")
            camera_worker.terminate()

        logic_worker.join(4) # Wait 2 seconds for logic_worker to gracefully terminate
        if logic_worker.exitcode is None:
            # the seg_worker subprocess did not terminate, try to terminate it forcefully
            print("Subprocess 'logic_worker' did not terminate properly, sending SIGTERM...")
            print("Not a big deal, just odd")
            logic_worker.terminate()

        # Add more worker cleanup here if needed
