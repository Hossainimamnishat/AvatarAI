"use client";
import React, { useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";

const Avatar = ({ avatarUrl }) => {
    const controlsRef = useRef();

    return (
        <div className="w-full h-[800px] flex justify-center items-center bg-gray-900 rounded-lg shadow-lg p-4">
            <Canvas 
                camera={{ position: [0, 2, 6], fov: 50 }} 
                className="w-full h-full"
                onCreated={({ gl }) => {
                    gl.toneMappingExposure = 1.3; // Improve lighting exposure
                }}
            >
                {/* Lights */}
                <ambientLight intensity={1.5} />
                <directionalLight position={[3, 5, 2]} intensity={2} />
                
                {/* Improved Orbit Controls */}
                <OrbitControls 
                    ref={controlsRef}
                    enableZoom={true} 
                    enablePan={true} 
                    enableRotate={true} 
                    minDistance={1.5} 
                    maxDistance={8} 
                    zoomSpeed={0.8} 
                    rotateSpeed={1} 
                    panSpeed={0.8}
                    maxPolarAngle={Math.PI}  // Allows full rotation
                    minPolarAngle={0}  // Prevents flipping upside-down
                    enableDamping={true}  // Smooth movement
                    dampingFactor={0.1}  // Makes interactions smoother
                />
                
                {/* Load the 3D model */}
                {avatarUrl && <Model url={avatarUrl} />}
            </Canvas>
        </div>
    );
};

const Model = ({ url }) => {
    const { scene } = useGLTF(url);
    return <primitive object={scene} scale={1.5} position={[0, -1, 0]} />;
};

export default Avatar;
