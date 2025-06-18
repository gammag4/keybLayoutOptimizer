import sys
import os
import json
from queue import Queue
import keyboard
import mouse


def safe_append_events_to_file(fname, new_events):
    # writes to temp file before writing to real file to prevent breaking in shutdown
    tmp = 'temp_' + fname

    if not os.path.exists(fname):
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump([], f)
        os.chown(fname, 0, 0)
        os.chmod(fname, 0o600)

    with open(fname, encoding='utf-8') as f:
        events = json.load(f)
    events = events + new_events

    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=4)
        # make sure that all data is on disk
        # see http://stackoverflow.com/questions/7433057/is-rename-without-fsync-safe
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, fname)  # os.rename pre-3.3, but os.rename won't work on Windows


def get_sort_existing_events(events_queue, events_to_write, last_events):
    sort_buffer_size = 50 # a buffer is kept to sort the last files just in case some event that happened before another arrives later in the queue

    events_to_write = events_to_write + last_events[:-sort_buffer_size]
    last_events = last_events[-sort_buffer_size:]

    while not events_queue.empty():
        e = events_queue.get()
        if type(e) == keyboard.KeyboardEvent:
            e = json.loads(e.to_json())
            e['keyboard_or_mouse'] = 'keyboard'
        else:
            e = { 'keyboard_or_mouse': 'mouse', 'time': e.time }

        print(e)
        last_events.append(e)

    last_events.sort(key=lambda e: e['time'])

    new_last = []
    last = events_to_write[-1] if len(events_to_write) > 0 else {'keyboard_or_mouse': ''}
    for i in last_events:
        # Prevents mouse events from filling all events, since the only thing that matters is whether the hand is going for the mouse or not
        if i['keyboard_or_mouse'] == 'mouse' and last['keyboard_or_mouse'] == 'mouse':
            continue
        new_last.append(i)
        last = i

    last_events = new_last

    return events_to_write, last_events


def main(filename):
    write_buffer_size = 100 # uses buffer just to prevent rewriting the whole file for every single event
    events_queue = Queue()
    events_to_write = [] # stores which events should be written next to file
    last_events = [] # stores a buffer with the last events found (see get_sort_existing_events for details)

    hook = lambda e: events_queue.put(e)

    keyboard.hook(hook)
    mouse.hook(hook)
    while True:
        events_to_write, last_events = get_sort_existing_events(events_queue, events_to_write, last_events)
        try:
            if len(events_to_write) > write_buffer_size:
                safe_append_events_to_file(filename, events_to_write)
                events_to_write = []
        except Exception as e:
            print(f'Error: {e}')


if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)
