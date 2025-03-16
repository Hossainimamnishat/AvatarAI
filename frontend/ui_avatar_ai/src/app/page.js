"use client";
import React, { useState } from "react";
import UploadForm from "../components/UploadForm";
import Avatar from "../components/Avatar";

export default function Home() {
    const [avatarUrl, setAvatarUrl] = useState("");
    const [gender, setGender] = useState("");

    return (
        <div className="flex flex-col items-center justify-center min-h-screen p-8 gap-8 bg-black text-white">
            <h1 className="text-3xl font-bold">3D Avatar Generator</h1>
            <UploadForm setAvatarUrl={setAvatarUrl} setGender={setGender} />
            {gender && <p className="text-lg">Detected Gender: <strong>{gender}</strong></p>}
            {avatarUrl && <Avatar avatarUrl={avatarUrl} />}
        </div>
    );
}
