from absl import app, flags
import roboverse

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", None, "Environment name.", required=True)

def main(_):
    env = roboverse.make(FLAGS.env_name, transpose_image=False, gui=True)
    for _ in range(10):
        env.reset()
        # input()
        # for _ in range(10):
        #     env.step([0, 0, -1, 0, 0])
        # input()

if __name__ == "__main__":
    app.run(main)