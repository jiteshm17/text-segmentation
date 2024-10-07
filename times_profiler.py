from timeit import default_timer as timer


class profiler:
    segments = []
    start = 0
    end = 0

    @staticmethod
    def set():
        """
        Mark the end of a segment and start the timer for the next segment.
        """
        profiler.end = timer()
        profiler.segments.append(profiler.end - profiler.start)
        profiler.start = timer()

    @staticmethod
    def init():
        """
        Initialize the profiler by starting the timer.
        """
        profiler.start = timer()

    @staticmethod
    def finish(profilerLog):
        """
        Finish profiling and log the results to the provided logger.
        
        Args:
            profilerLog: A logger object to which profiling results will be logged.
        """
        profiler.end = timer()
        profiler.segments.append(profiler.end - profiler.start)

        # Format the results for logging
        str2log = " ".join([f"{i}-{i+1} = {segment:.2f}" for i, segment in enumerate(profiler.segments)])
        profilerLog.debug(str2log)

        # Clear the segments after logging
        profiler.segments = []