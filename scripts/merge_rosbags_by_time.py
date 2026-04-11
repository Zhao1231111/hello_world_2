#!/usr/bin/env python3
"""Merge multiple ROS bag files into one output bag ordered by message time."""

import argparse
import heapq
import os
import sys

import rosbag


def positive_float(value):
    seconds = float(value)
    if seconds < 0:
        raise argparse.ArgumentTypeError("Duration must be non-negative.")
    return seconds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple input rosbags into one output bag ordered by timestamp."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output bag path.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input bag paths.",
    )
    parser.add_argument(
        "--topics",
        nargs="*",
        default=None,
        help="Optional topic whitelist. When omitted, all topics are copied.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("bag", "header"),
        default="bag",
        help="Sort merged output by rosbag record time or by message header.stamp.",
    )
    parser.add_argument(
        "--max-duration",
        type=positive_float,
        default=None,
        help=(
            "Only merge the first N seconds starting from the earliest message in the "
            "merged output timeline."
        ),
    )
    return parser.parse_args()


def get_sort_stamp(msg, bag_stamp, sort_by):
    if sort_by == "bag":
        return bag_stamp

    if hasattr(msg, "header"):
        return msg.header.stamp

    raise ValueError(
        f"Message type {type(msg).__name__} has no header; cannot sort by header timestamp."
    )


def push_next(heap, iterators, bag_idx, counter, sort_by):
    iterator = iterators[bag_idx]
    try:
        topic, msg, stamp = next(iterator)
    except StopIteration:
        return counter

    sort_stamp = get_sort_stamp(msg, stamp, sort_by)
    heapq.heappush(
        heap,
        (sort_stamp.to_nsec(), counter, bag_idx, topic, msg, stamp, sort_stamp),
    )
    return counter + 1


def main():
    args = parse_args()

    if os.path.abspath(args.output) in [os.path.abspath(path) for path in args.inputs]:
        raise ValueError("Output bag must be different from all input bags.")

    input_bags = []
    iterators = []
    heap = []
    counter = 0
    total_written = 0

    try:
        for path in args.inputs:
            bag = rosbag.Bag(path, "r")
            input_bags.append(bag)
            iterator = bag.read_messages(topics=args.topics)
            iterators.append(iterator)
            counter = push_next(
                heap, iterators, len(iterators) - 1, counter, args.sort_by
            )

        if not heap:
            raise RuntimeError("No messages matched the requested topics.")

        max_end_time_ns = None
        if args.max_duration is not None:
            max_end_time_ns = heap[0][0] + int(args.max_duration * 1e9)

        with rosbag.Bag(args.output, "w") as outbag:
            while heap:
                _, _, bag_idx, topic, msg, stamp, sort_stamp = heapq.heappop(heap)
                if max_end_time_ns is not None and sort_stamp.to_nsec() > max_end_time_ns:
                    break
                write_stamp = sort_stamp if args.sort_by == "header" else stamp
                outbag.write(topic, msg, write_stamp)
                total_written += 1
                if total_written % 10000 == 0:
                    print(f"[merge_rosbags_by_time] written {total_written} messages", flush=True)
                counter = push_next(
                    heap, iterators, bag_idx, counter, args.sort_by
                )

    finally:
        for bag in input_bags:
            bag.close()

    if args.max_duration is None:
        print(f"[merge_rosbags_by_time] wrote {total_written} messages to {args.output}")
    else:
        print(
            f"[merge_rosbags_by_time] wrote {total_written} messages to {args.output} "
            f"(first {args.max_duration} seconds)"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[merge_rosbags_by_time] error: {exc}", file=sys.stderr)
        sys.exit(1)
