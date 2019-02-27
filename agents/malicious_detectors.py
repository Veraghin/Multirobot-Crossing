import numpy as np


def oracle(malicious):
    def identifier(neighbours):
        return [n for n in neighbours if n.id == malicious]

    return identifier


def history(threshold):
    historical_headings = dict()

    # Controller tries to find the malicious robot based on it's heading
    # Assumes no gap in communication between robots
    def identifier(neighbours):
        enough_data = True
        if len(neighbours) == 0:
            enough_data = False
        for n in neighbours:
            name = n.id
            heading = n.get_velocity()
            if name in historical_headings.keys():
                historical_headings[name].append(heading)
                if len(historical_headings[name]) < 20:
                    enough_data = False
            else:
                historical_headings[name] = [heading]
                enough_data = False
        # If there is a sufficient heading information available
        malicious = []
        if enough_data:
            for n, headings in historical_headings.items():
                change = 0.0
                for i in range(len(headings) - 1):
                    change += np.linalg.norm(headings[i + 1] - headings[i])
                if change / (len(headings) - 1) < threshold:
                    malicious.append(n)
        return [n for n in neighbours if n.id in malicious]
    return identifier
