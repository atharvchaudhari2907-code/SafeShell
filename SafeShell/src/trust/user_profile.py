<<<<<<< HEAD
# Personalization Module
# Member 5 - AI-Based Safe Linux Command Execution

USER_PROFILES = {
    "normal": {
        "name": "Normal User",
        "risk_level": "strict"
    },

    "developer": {
        "name": "Developer",
        "risk_level": "moderate"
    },

    "admin": {
        "name": "Administrator",
        "risk_level": "high"
    }
}


def get_user_profile(role):
    """Return the policy profile for a user role."""

    role = role.lower().strip()

    if role in USER_PROFILES:
        return USER_PROFILES[role]

    return USER_PROFILES["normal"]


if __name__ == "__main__":
    role = input("Enter user role (normal/developer/admin): ")

    profile = get_user_profile(role)

    print("User:", profile["name"])
=======
# Personalization Module
# Member 5 - AI-Based Safe Linux Command Execution

USER_PROFILES = {
    "normal": {
        "name": "Normal User",
        "risk_level": "strict"
    },
    "developer": {
        "name": "Developer",
        "risk_level": "moderate"
    },
    "admin": {
        "name": "Administrator",
        "risk_level": "high"
    }
}


def get_user_profile(role):
    """Return the policy profile for a user role."""
    role = role.lower().strip()
    if role in USER_PROFILES:
        return USER_PROFILES[role]
    return USER_PROFILES["normal"]


if __name__ == "__main__":
    role = input("Enter user role (normal/developer/admin): ")
    profile = get_user_profile(role)
    print("User:", profile["name"])
>>>>>>> c3555bea6ec162adf3bd477996aab1afb5b3039e
    print("Risk Policy:", profile["risk_level"])