#pragma once

#define GLM_FORCE_LEFT_HANDED
#include <glm/glm.hpp>
#include <SDL2/SDL.h>

class Camera
{
public:
    Camera(float _fov, float _aspect, float _near, float _far, const glm::vec3& _position = glm::vec3(0.0f, 0.0f, -3.0f));

    // Call this in your SDL event loop
    void HandleEvent(const SDL_Event& e);

    // Call this once per frame with the delta time
    void Update(float _dt);

    // Getters
    const glm::mat4& GetViewMatrix() const { return m_viewMat; }
    const glm::mat4& GetProjMatrix() const { return m_projMat; }

private:
    void UpdateMatrices();

private:
    // Matrices
    glm::mat4 m_viewMat;
    glm::mat4 m_projMat;

    // Camera vectors
    glm::vec3 m_position;
    glm::vec3 m_front;
    glm::vec3 m_up;
    glm::vec3 m_right;
    glm::vec3 m_worldUp;

    // Euler angles
    float m_yaw;
    float m_pitch;

    // Settings
    float m_speed;
    float m_sensitivity;
    float m_fov;
    float m_aspect;
    float m_near;
    float m_far;

    // State
    bool m_isLooking;
};