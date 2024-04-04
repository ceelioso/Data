import React, {useEffect, useState} from "react";
import {useHistory} from "react-router-dom";
import axios from "axios";

const ProfilePage = () => {
    const [user, setUser] = useState({});
    const [sessionUrl,] = useState("/api/sessions/me");
    const history = useHistory();

    useEffect(() => {
         (async () => {
            try {
                const response = await axios.get(sessionUrl);
                setUser(response.data);
            } catch (err) {
                history.push({
                    pathname: "/account/login"
                });
            }
        })();
     }, []);

     return (
         <div>
             <div>Name: {user.displayName}</div>
             <div>Username: {user.username}</div>
             <div>Email: {user.email}</div>
         </div>
    );
 }

export default ProfilePage;