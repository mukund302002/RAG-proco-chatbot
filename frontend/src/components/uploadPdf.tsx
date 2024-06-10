/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/ban-ts-comment */
import axios from "axios";
import { useState } from "react";
//@ts-ignore
const UploadPDF = ({ setPdfId }) => {
  const [file, setFile] = useState(null);
  const [error, setError] = useState("");
  const [customId, setCustomId] = useState("");

  const handleFileChange = (e: any) => {
    setFile(e.target.files[0]);
    setError("");
  };

  const handleIdChange = (e: any) => {
    setCustomId(e.target.value);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first.");
      return;
    }

    if (!customId) {
      setError("Please enter a custom ID.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("id", customId);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setPdfId(customId);
    } catch (error) {
      setError("Failed to upload PDF.");
    }
  };

  return (
    <div>
      <h2>Upload PDF</h2>
      <input type="file" onChange={handleFileChange} />
      <input
        type="text"
        placeholder="Enter custom ID"
        value={customId}
        onChange={handleIdChange}
      />
      <button onClick={handleUpload}>Upload</button>
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
};

export default UploadPDF;
