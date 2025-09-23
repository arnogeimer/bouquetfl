from nvitop import Device, ResourceMetricCollector, collect_in_background

resources = []
global stop
stop = False


def on_collect(metrics):
    if stop:  # closed manually by user
        return False
    resources.append(metrics)
    return True


def on_stop(collector):  # will be called only once at stop
    if stop:
        raise SystemExit(0)  # cleanup


def start_collection():
    collect_in_background(
        on_collect,
        ResourceMetricCollector(Device.cuda.all()),
        interval=0.5,
        on_stop=on_stop,
    )


def stop_collection():
    stop = True
    print("Stopped resource collection.")
    return resources
