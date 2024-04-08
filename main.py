from video import threadVideo

def main():
    xyz = threadVideo()
    xyz.start()

    while True:
        # TODO GAME LOGIC
        print(xyz.dist)

if __name__ == '__main__':
    main()