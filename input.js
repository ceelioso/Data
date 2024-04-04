import React, { memo, useState, useEffect } from "react";
import axios from "axios";

import "./Input.css";

const Input = memo(({ placeholder, type, value, handleInput, name }) => ( <
    input type = { type }
    name = { name }
    value = { value }
    onChange = { handleInput }
    placeholder = { placeholder }
    />
));

export default Input;

// Define INITIAL_STATE outside of the component to avoid redeclaration
const INITIAL_STATE = { id: null, name: "", email: "" };

// Define your component (assuming App is the main component)
const App = () => {
    const [userData, setUserData] = useState(INITIAL_STATE);

    useEffect(() => {
        const fetchUserData = async() => {
            try {
                if (!userData.id) return; // Prevent null pointer reference
                const { data } = await axios.get(
                    `https://yourendpoint/${userData.id}`
                );
                setUserData(data);
            } catch (error) {
                console.error("Error fetching user data:", error);
            }
        };

        fetchUserData();
    }, [userData.id]);

    const handleInputChange = ({ target: { name, value } }) =>
        name && value && setUserData(prevState => ({...prevState, [name]: value }));

    const handleFormSubmit = async event => {
        event.preventDefault();

        try {
            if (!userData.id) {
                console.error("User data does not have an ID");
                return;
            }
            await axios.put(`https://yourendpoint/${userData.id}`, userData);
        } catch (error) {
            console.error("Error updating user data:", error);
        }
    };

    return ( <
        div className = "App" >
        <
        h1 > { userData.name || "Loading..." } < /h1> <
        form onSubmit = { handleFormSubmit } >
        <
        Input name = "name"
        type = "text"
        value = { userData.name || "" }
        placeholder = "Your name"
        handleInput = { handleInputChange }
        /> <
        br / >
        <
        Input name = "email"
        type = "email"
        value = { userData.email || "" }
        placeholder = "Your email"
        handleInput = { handleInputChange }
        /> <
        br / >
        <
        input type = "submit"
        value = "Update"
        disabled = {!userData.id }
        /> <
        /form> <
        /div>
    );
};