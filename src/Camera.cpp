#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(float _fov, float _aspect, float _near, float _far, const glm::vec3& _position)
    : m_fov(_fov), m_aspect(_aspect), m_near(_near), m_far(_far), m_position(_position),
      m_yaw(90.0f), m_pitch(0.0f), m_speed(3.0f), m_sensitivity(0.1f), m_isLooking(false)
{
    m_worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
    UpdateMatrices();
}

void Camera::HandleEvent(const SDL_Event& e)
{
    if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_RIGHT)
    {
        m_isLooking = true;
        SDL_SetRelativeMouseMode(SDL_TRUE); // Hide and lock cursor
    }
    else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_RIGHT)
    {
        m_isLooking = false;
        SDL_SetRelativeMouseMode(SDL_FALSE); // Show and unlock cursor
    }
    else if (e.type == SDL_MOUSEMOTION && m_isLooking)
    {
        // e.motion.xrel and yrel are the change in mouse position since last frame
        m_yaw -= e.motion.xrel * m_sensitivity;
        m_pitch -= e.motion.yrel * m_sensitivity;

        // Clamp pitch to prevent flipping the camera upside down
        if (m_pitch > 89.0f) m_pitch = 89.0f;
        if (m_pitch < -89.0f) m_pitch = -89.0f;

        UpdateMatrices();
    }
    else if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_RESIZED)
    {
        m_aspect = static_cast<float>(e.window.data1) / static_cast<float>(e.window.data2);
        UpdateMatrices();
    }
}

void Camera::Update(float _dt)
{
    if (!m_isLooking) return; // Only move if we are holding right click

    const Uint8* keyboardState = SDL_GetKeyboardState(nullptr);
    float velocity = m_speed * _dt;

    // Standard FPS "Fly" camera controls
    if (keyboardState[SDL_SCANCODE_W])
        m_position += m_front * velocity;
    if (keyboardState[SDL_SCANCODE_S])
        m_position -= m_front * velocity;
    if (keyboardState[SDL_SCANCODE_A])
        m_position -= m_right * velocity;
    if (keyboardState[SDL_SCANCODE_D])
        m_position += m_right * velocity;
    
    // Optional: Up/Down with Space/Shift
    if (keyboardState[SDL_SCANCODE_SPACE])
        m_position += m_worldUp * velocity;
    if (keyboardState[SDL_SCANCODE_LSHIFT])
        m_position -= m_worldUp * velocity;

    UpdateMatrices();
}

void Camera::UpdateMatrices()
{
    // 1. Calculate the new Front vector using Euler angles
    glm::vec3 front;
    front.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    front.y = sin(glm::radians(m_pitch));
    front.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    m_front = glm::normalize(front);

    // 2. Calculate the Right and Up vectors
    m_right = glm::normalize(glm::cross(m_worldUp, m_front));
    m_up    = glm::normalize(glm::cross(m_front, m_right));

    // 3. Update Matrices
    m_viewMat = glm::lookAt(m_position, m_position + m_front, m_up);
    m_projMat = glm::perspective(glm::radians(m_fov), m_aspect, m_near, m_far);
}