
import time
import threading
import logging
from pose_analyzer.concurrent_stream_processor import ConcurrentStreamProcessor, SyntheticFrameSource
from pose_analyzer.pose_detector import YOLOPosev11Detector

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("StreamManagerTest")

    logger.info("Initializing Detector...")
    detector = YOLOPosev11Detector(model_variant="n", imgsz=640)
    
    logger.info("Initializing Processor...")
    processor = ConcurrentStreamProcessor(detector, max_queue_size=32)
    
    # Create 4 sources with 300 frames each (10 sec @ 30fps)
    sources = [
        SyntheticFrameSource(num_frames=600, target_fps=30) 
        for _ in range(4)
    ]
    
    # Run processor in background thread
    logger.info("Starting Processor in background thread...")
    proc_thread = threading.Thread(
        target=processor.run, 
        kwargs={'sources': sources, 'collect_results': True}
    )
    proc_thread.start()
    
    # Monitor and Manage
    try:
        start_time = time.monotonic()
        while proc_thread.is_alive():
            elapsed = time.monotonic() - start_time
            
            # Print status every second
            stats = []
            for i in range(4):
                m = processor.stream_metrics.get(i)
                processed = m.frames_processed if m else 0
                dropped = m.frames_dropped if m else 0
                paused = "PAUSED" if i in processor.stream_paused_events and not processor.stream_paused_events[i].is_set() else "RUNNING"
                stats.append(f"S{i}:{paused} (P:{processed}/D:{dropped})")
            
            logger.info(f"Status: {', '.join(stats)}")

            # SCENARIO: At t=3s, Pause Stream 3 (Simulate VRAM Pressure)
            if 3.0 <= elapsed < 4.0 and processor.stream_paused_events.get(3).is_set():
                logger.warning(">>> SIMULATING HIGH VRAM! Pausing Stream 3...")
                processor.pause_stream(3)
            
            # SCENARIO: At t=6s, Resume Stream 3 (Simulate VRAM Recovery)
            if 6.0 <= elapsed < 7.0 and not processor.stream_paused_events.get(3).is_set():
                logger.info(">>> VRAM RECOVERED. Resuming Stream 3...")
                processor.resume_stream(3)

            # Stop after 10s
            if elapsed > 10.0:
                logger.info("Test complete. Stopping...")
                processor._stop_event.set()
                break
                
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        processor._stop_event.set()
        
    proc_thread.join()
    logger.info("Done.")

if __name__ == "__main__":
    main()
