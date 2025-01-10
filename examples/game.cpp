#include <raylib.h>
#include <cmath>
#include <string>
#include "network.h"
#include "layers/fc_layer.h"
#include "layers/activation_layer.h"
#include <vector>
#include <random>

using namespace Neural;

const int screenWidth = 800;
const int screenHeight = 600;
const float playerSize = 20.0f;
const float playerSpeed = 2.0f;
const float playerRotationSpeed = 2.0f;
const float wallThickness = 10.0f;

#define MOVE_LEFT     0x01
#define MOVE_RIGHT    0x02
#define MOVE_FORWARD  0x04
#define MOVE_BACKWORD 0x08

struct Player
{
  Vector2 position;
  float rotation;
  Vector2 direction;
};

struct Goal
{
  Vector2 position;
  float radius;
};

Player player;
Goal goal;

void InitGame()
{
  player.position = (Vector2){screenWidth / 2.0f, screenHeight / 2.0f};
  player.rotation = (float)GetRandomValue(0, 360);
  player.direction = (Vector2){1.0f, 0.0f};

  goal.position = (Vector2){(float)GetRandomValue(50, screenWidth - 50), (float)GetRandomValue(50, screenHeight - 50)};
  goal.radius = 10.0f;
}

void UpdatePlayer(int inputAction)
{
  if (inputAction & MOVE_FORWARD)
  {
    player.position.x += player.direction.x * playerSpeed;
    player.position.y += player.direction.y * playerSpeed;
  }
  else if (inputAction & MOVE_BACKWORD)
  {
    player.position.x -= player.direction.x * playerSpeed;
    player.position.y -= player.direction.y * playerSpeed; 
  }

  if (inputAction & MOVE_LEFT)
    player.rotation += playerRotationSpeed;
  if (inputAction & MOVE_RIGHT)
    player.rotation -= playerRotationSpeed;

  // Update direction
  player.direction.x = cos(player.rotation * DEG2RAD);
  player.direction.y = sin(player.rotation * DEG2RAD);

  // Collision Detection
  if (player.position.x < wallThickness)
    player.position.x = wallThickness;
  if (player.position.y < wallThickness)
    player.position.y = wallThickness;
  if (player.position.x > screenWidth - wallThickness)
    player.position.x = screenWidth - wallThickness;
  if (player.position.y > screenHeight - wallThickness)
    player.position.y = screenHeight - wallThickness;
}

bool CheckGoalReached()
{
  return CheckCollisionCircleRec(goal.position, goal.radius,
                                 (Rectangle){player.position.x - playerSize / 2, player.position.y - playerSize / 2, playerSize, playerSize});
}

void CalculateDirectionToGoal(float &angleDifference, float &distance)
{
  float dx = goal.position.x - player.position.x;
  float dy = goal.position.y - player.position.y;

  // Caculate angle
  float angleToGoal = atan2(dy, dx) * RAD2DEG;

  // Convert angle to 0-360 degree range
  if (angleToGoal < 0) angleToGoal += 360;

  angleDifference = angleToGoal - fmod(player.rotation, 360);

  if (angleDifference > 180)  angleDifference -= 360;
  if (angleDifference < -180) angleDifference += 360;

  // Caculate distance
  distance = sqrt(dx * dx + dy * dy);
}

void DrawGame()
{
  // Draw walls
  DrawRectangle(0, 0, screenWidth, wallThickness, GRAY);
  DrawRectangle(0, screenHeight - wallThickness, screenWidth, wallThickness, GRAY);
  DrawRectangle(0, 0, wallThickness, screenHeight, GRAY);
  DrawRectangle(screenWidth - wallThickness, 0, wallThickness, screenHeight, GRAY);

  // 繪製玩家
  DrawRectanglePro(
      (Rectangle){player.position.x, player.position.y, playerSize, playerSize},
      (Vector2){playerSize / 2, playerSize / 2},
      player.rotation,
      BLUE);

  // 繪製玩家方向
  DrawLineEx(player.position,
             (Vector2){player.position.x + player.direction.x * 30, player.position.y + player.direction.y * 30},
             3, RED);

  // 繪製目標
  DrawCircleV(goal.position, goal.radius, GREEN);
}



class QNetwork : public Neural::Network
{
public:
  QNetwork() : Neural::Network()
  {
    // Input layer: 2 neurons (angle, distance)
    this->Add(new Fc_Layer(2, 64, ActivationType::TANH));
    this->Add(new Fc_Layer(64, 32, ActivationType::LEAKY_RELU));
    // Output layer: 3 neurons (left, right, none)
    this->Add(new Fc_Layer(32, 3, ActivationType::NONE));

    this->Use(new Mse());
  }

  Eigen::MatrixXd GetQValues(const Eigen::MatrixXd &state)
  {
    return this->Predict(state)[0];
  }
};



struct Experience {
    Eigen::MatrixXd state;
    int action;
    double reward;
    Eigen::MatrixXd next_state;
    bool done;
};

class ReplayBuffer {
private:
    std::vector<Experience> buffer;
    size_t max_size;
    std::random_device rd;
    std::mt19937 gen;

public:
    ReplayBuffer(size_t size) : max_size(size), gen(rd()) {}

    void add(const Experience& exp) {
        if (buffer.size() >= max_size) {
            buffer.erase(buffer.begin());
        }
        buffer.push_back(exp);
    }

    std::vector<Experience> sample(size_t batch_size) {
        std::vector<Experience> batch;
        for (size_t i = 0; i < batch_size && i < buffer.size(); ++i) {
            size_t index = std::uniform_int_distribution<>(0, buffer.size() - 1)(gen);
            batch.push_back(buffer[index]);
        }
        return batch;
    }

    size_t size() const {
        return buffer.size();
    }
};


class QLearningAgent
{
private:
  QNetwork qnetwork;
  double epsilon;
  double gamma;
  double learning_rate;

  ReplayBuffer replay_buffer;
  size_t batch_size;

public:
  QLearningAgent(double eps = 1.0, double g = 0.99, double lr = 0.001, size_t buffer_size = 10000, size_t batch = 320)
      : epsilon(eps), gamma(g), learning_rate(lr), replay_buffer(buffer_size), batch_size(batch) {}

  int GetAction(const Eigen::MatrixXd &state)
  {
    // if (((double)rand() / RAND_MAX) < epsilon)
    // {
    //   if (epsilon > 0) epsilon -= 0.0001;
    //   return rand() % 3; // 隨機探索
    // }
    // else
    {
      Eigen::MatrixXd q_values = qnetwork.GetQValues(state);
      Eigen::MatrixXd::Index maxRow, maxCol;
      q_values.maxCoeff(&maxRow, &maxCol);
      return static_cast<int>(maxCol);
    }
  }

  void Update(const Eigen::MatrixXd &state, int action, double reward, const Eigen::MatrixXd &next_state, bool done)
  {
    // 添加經驗到緩衝區
    replay_buffer.add({state, action, reward, next_state, done});

    // 如果緩衝區中的經驗不夠，就不進行學習
    if (replay_buffer.size() < batch_size) return;

    auto batch = replay_buffer.sample(batch_size);
    batch.push_back({state,action, reward, next_state, done});

    for (const auto &exp : batch) {
      Eigen::MatrixXd q_values = qnetwork.GetQValues(exp.state);
      Eigen::MatrixXd next_q_values = qnetwork.GetQValues(exp.next_state);

      double target = exp.reward;
      if (!exp.done) {
        target += gamma * next_q_values.maxCoeff();
      }

      q_values(exp.action) = (1 - learning_rate) * q_values(exp.action) + learning_rate * target;

      Eigen::MatrixXd input(1, 2);
      input << exp.state;
      Eigen::MatrixXd output(1, 3);
      output << q_values;

      qnetwork.Fit(input, output, 1, learning_rate, 0);
    }
  }

  void SaveModel(void) {
    qnetwork.SaveModel("test");
  }
};


int main()
{
  InitWindow(screenWidth, screenHeight, "AI Training Environment");
  SetTargetFPS(60);

  RenderTexture2D renderTexture = LoadRenderTexture(screenWidth, screenHeight);

  InitGame();

  QLearningAgent agent;
  double reward = 0;
  double time = 0;
  float angleDifference, distanceToGoal;

  while (!WindowShouldClose())
  {
    // Get current state
    CalculateDirectionToGoal(angleDifference, distanceToGoal);
    Eigen::MatrixXd state(1, 2);
    state << angleDifference, distanceToGoal;

    // Get action
    int action = agent.GetAction(state);
    int inputAction = 0;

    if (IsKeyDown(KEY_LEFT))  action = 1;
    if (IsKeyDown(KEY_RIGHT)) action = 2;

    if (action == 0)      inputAction = 0;
    else if (action == 1) inputAction = MOVE_LEFT;
    else if (action == 2) inputAction = MOVE_RIGHT;

    UpdatePlayer(inputAction | MOVE_FORWARD);

    bool done = false;
    if (CheckGoalReached())
    {
      done = true;
      reward = 100;
      InitGame();
    }
    else if (player.position.x <= wallThickness || player.position.x >= screenWidth - wallThickness ||
             player.position.y <= wallThickness || player.position.y >= screenHeight - wallThickness)
    {
      done = true;
      reward = -10; // 撞牆懲罰
      InitGame();
    }

    // Get new state
    float newAngleDifference, newDistanceToGoal;
    CalculateDirectionToGoal(newAngleDifference, newDistanceToGoal);

    if (!done) {
      time += 0.01;
      reward = -time;
      // 計算獎勵
      if (fabs(newAngleDifference) < 10) {
        reward += 0.5;
      }
      else if (fabs(newAngleDifference) < fabs(angleDifference)) 
        reward += 0;
      else
        reward += -20;
      
      if (newDistanceToGoal < distanceToGoal)
        reward += 0.1;
      else
        reward += -1;
    }
    else {
      time = 0;
    }

    Eigen::MatrixXd next_state(1, 2);

    next_state << newAngleDifference, newDistanceToGoal;

    // 更新Q-network
    agent.Update(state, action, reward, next_state, done);

    if (done) reward = 0;

    BeginTextureMode(renderTexture);
      ClearBackground(RAYWHITE);
      DrawGame();
    EndTextureMode();

    BeginDrawing();
      ClearBackground(RAYWHITE);
      DrawTextureEx(renderTexture.texture, (Vector2){0, 0}, 0, 1.0, RAYWHITE);
      DrawText(("Reward: " + std::to_string(static_cast<double>(reward))).c_str(), 10, 70, 20, RED);
      DrawText(("Angle to Goal: " + std::to_string(static_cast<int>(angleDifference)) + "°").c_str(), 10, 10, 20, BLACK);
      DrawText(("Distance to Goal: " + std::to_string(static_cast<int>(distanceToGoal))).c_str(), 10, 40, 20, BLACK);      
    EndDrawing();
  }

  agent.SaveModel();

  CloseWindow();
  return 0;
}