#pragma once

#include <unordered_set>
#include <glm/glm.hpp>

struct InputState {
	bool isKeyPressed(int keyCode) const {
		return m_pressedKeys.count(keyCode) > 0;
	}

	bool isMouseButtonPressed(int mouseButtonCode) const {
		return m_pressedMouseButtons.count(mouseButtonCode) > 0;
	}

	void setKeyPressed(int keyCode, bool bPressed) {
		if (bPressed) {
			m_pressedKeys.insert(keyCode);
		}
		else {
			m_pressedKeys.erase(keyCode);
		}
	}

	void setMouseButtonPressed(int mouseButtonCode, bool bPressed) {
		if (bPressed) {
			m_pressedMouseButtons.insert(mouseButtonCode);
		}
		else {
			m_pressedMouseButtons.erase(mouseButtonCode);
		}
	}

	std::unordered_set<int> m_pressedKeys;
	std::unordered_set<int> m_pressedMouseButtons;
	glm::vec2 m_mousePosition = glm::vec2(0.0f);
	glm::vec2 m_scrollOffset = glm::vec2(0.0f);
};