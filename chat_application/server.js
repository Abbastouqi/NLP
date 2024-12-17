// Import necessary modules
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');

// Create an Express app and an HTTP server
const app = express();
const server = http.createServer(app);
const io = socketIo(server);

// Serve static files from the "public" directory
app.use(express.static('public'));

// Set up a connection event for Socket.IO
io.on('connection', (socket) => {
    console.log('A user connected');

    // Listen for 'chat message' event from the client
    socket.on('chat message', (msg) => {
        // Broadcast the message to all clients
        io.emit('chat message', msg);
    });

    // Handle user disconnection
    socket.on('disconnect', () => {
        console.log('User disconnected');
    });
});

// Start the server
server.listen(3000, () => {
    console.log('Server is listening on port 3000');
});
