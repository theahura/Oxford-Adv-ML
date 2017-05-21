"""
Author: Amol Kapoor
Description: Runs the mnist gan
"""

import load_data


def main():
    data = load_data.get_data()
    print data.train.next_batch(100)

if __name__ == '__main__':
    main()
