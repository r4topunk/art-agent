// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/// @title ArtAgent — Autonomous Pixel Art NFT Collection
/// @notice One piece minted per evolutionary generation. Owner-only minting.
contract ArtAgent is ERC721, ERC721URIStorage, Ownable {
    uint256 private _nextTokenId;

    /// @notice Emitted when a new generation piece is minted.
    event GenerationMinted(
        uint256 indexed tokenId,
        uint256 indexed generation,
        string tokenURI
    );

    /// @notice Maps tokenId → generation number.
    mapping(uint256 => uint256) public tokenGeneration;

    constructor(
        address initialOwner
    ) ERC721("ArtAgent", "ARTAGENT") Ownable(initialOwner) {}

    /// @notice Mint a new piece for a given generation.
    /// @param to      Recipient address.
    /// @param generation  The evolutionary generation number.
    /// @param uri     IPFS metadata URI (ipfs://...).
    /// @return tokenId The newly minted token ID.
    function mint(
        address to,
        uint256 generation,
        string calldata uri
    ) external onlyOwner returns (uint256) {
        uint256 tokenId = _nextTokenId++;
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, uri);
        tokenGeneration[tokenId] = generation;
        emit GenerationMinted(tokenId, generation, uri);
        return tokenId;
    }

    /// @notice Total number of minted tokens.
    function totalSupply() external view returns (uint256) {
        return _nextTokenId;
    }

    // --- Required overrides ---

    function tokenURI(
        uint256 tokenId
    ) public view override(ERC721, ERC721URIStorage) returns (string memory) {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(
        bytes4 interfaceId
    ) public view override(ERC721, ERC721URIStorage) returns (bool) {
        return super.supportsInterface(interfaceId);
    }
}
