"use client";
import React, { useState } from "react";
import axios from "axios";

const UploadForm = ({ setAvatarUrl, setGender }) => {
    const [file, setFile] = useState(null);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        console.log("Selected file:", e.target.files[0]);
    };

    const handleUpload = async () => {
        if (!file) {
            console.error("No file selected");
            return;
        }

        const formData = new FormData();
        formData.append("front", file);

        try {
            console.log("Uploading file...");
            const response = await axios.post("http://localhost:8000/upload/", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            console.log("Upload Response:", response.data);
            setAvatarUrl(response.data.avatar_url);
            setGender(response.data.gender);

        } catch (error) {
            console.error("Upload failed:", error);
        }
    };

    return (
        <div className="flex flex-col gap-4">
            <input type="file" onChange={handleFileChange} className="p-2 border" />
            <button onClick={handleUpload} className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                Upload Image
            </button>
        </div>
    );
};

export default UploadForm;
