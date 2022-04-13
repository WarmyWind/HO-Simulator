import sys
import time

def search_object_form_list_by_no(object_list, no):
    for _obj in object_list:
        if _obj.no == no:
            return _obj


def progress_bar(pct):
    print("\r", end="")
    print("Simulation Progress: {:.2f}%: ".format(pct), "▋" * int(pct // 2), end="")
    if pct != 100:
        sys.stdout.flush()
    else:
        print('\n')